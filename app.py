from datetime import date
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.auth import require_login
from utils.data_loader import load_data
from utils.config import BLOOD_GROUPS, STOCK_STATUSES, METRICS_PATH, CLASSIFIER_PATH, DEMAND_MODEL_PATH
from ml.stock_status_model import (
    FEATURES, train_stock_status_model, load_stock_status_model, predict_stock_status
)
from ml.demand_forecasting import (
    train_demand_model, load_demand_model, forecast_demand
)
from utils.operations import init_operations, inventory as get_inventory, donations as get_donations, requests as get_requests, add_donation, add_request

st.set_page_config(page_title="BloodBank AI", page_icon="🩸", layout="wide")
user = require_login()

@st.cache_resource
def get_classifier():
    if not CLASSIFIER_PATH.exists():
        train_stock_status_model(data)
    return load_stock_status_model()

@st.cache_resource
def get_demand_model():
    if not DEMAND_MODEL_PATH.exists():
        train_demand_model(data)
    return load_demand_model()

try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

init_operations(data)

banks = (data[["blood_bank_id", "blood_bank_name", "city", "state"]]
         .drop_duplicates().sort_values("blood_bank_name"))
if user["role"] == "super_admin":
    with st.sidebar:
        st.title("🩸 BloodBank AI")
        st.caption("Multi-Blood-Bank Intelligence")
        bank_options = ["All blood banks"] + banks["blood_bank_id"].tolist()
        selected = st.selectbox(
            "Select blood bank", bank_options,
            format_func=lambda x: "All blood banks" if x == "All blood banks"
            else banks.loc[banks.blood_bank_id == x, "blood_bank_name"].iloc[0]
        )
        page = st.radio("Navigation", [
            "Dashboard", "Inventory", "Record Donation", "Blood Requests",
            "Demand Forecasting", "Shortage Prediction", "Blood Bank Network",
            "Dataset Explorer", "Model Performance", "Reports"
        ])
        st.success("Super Admin")
else:
    selected = user["bank_id"]
    with st.sidebar:
        st.title("🩸 BloodBank AI")
        bank_name = banks.loc[banks.blood_bank_id == selected, "blood_bank_name"].iloc[0]
        st.info(bank_name)
        page = st.radio("Navigation", [
            "Dashboard", "Inventory", "Record Donation", "Blood Requests",
            "Demand Forecasting", "Shortage Prediction", "Dataset Explorer", "Model Performance", "Reports"
        ])
        st.caption("Bank Admin")

if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.pop("user", None)
    st.cache_resource.clear()
    st.rerun()

filtered = data if selected == "All blood banks" else data[data["blood_bank_id"] == selected]
latest_date = data["date"].max()
latest = filtered[filtered["date"] == latest_date]
if latest.empty:
    latest = filtered[filtered["date"] == filtered["date"].max()]

def title(heading, sub):
    st.title(heading)
    st.caption(sub)

def bank_id_for_model():
    if selected == "All blood banks":
        return st.selectbox("Blood bank", banks["blood_bank_id"].tolist(),
            format_func=lambda x: banks.loc[banks.blood_bank_id == x, "blood_bank_name"].iloc[0])
    return selected

if page == "Dashboard":
    title("BloodBank AI Dashboard", "A data-driven overview of blood demand and stock status.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Dataset records", f"{len(filtered):,}")
    c2.metric("Blood banks", filtered["blood_bank_id"].nunique())
    c3.metric("Blood groups", filtered["blood_group"].nunique())
    c4.metric("Latest data date", latest_date.strftime("%d %b %Y"))
    st.subheader("Latest stock status distribution")
    status_counts = latest["stock_status"].value_counts().reindex(STOCK_STATUSES, fill_value=0).reset_index()
    status_counts.columns = ["Stock status", "Records"]
    left,right = st.columns([1,1])
    left.plotly_chart(px.pie(status_counts, names="Stock status", values="Records",
                             hole=0.48, title="Latest recorded status"), use_container_width=True)
    trend = filtered.groupby(["date","stock_status"]).size().reset_index(name="records")
    right.plotly_chart(px.area(trend, x="date", y="records", color="stock_status",
                               title="Daily stock-status history"), use_container_width=True)
    st.subheader("Latest inventory snapshot")
    cols = ["blood_bank_name","city","blood_group","opening_stock_units",
            "donations_received_units","total_demand_units","closing_stock_units","stock_status"]
    st.dataframe(latest[cols].sort_values(["blood_bank_name","blood_group"]),
                 use_container_width=True, hide_index=True)
    st.subheader("Demand over time")
    demand = filtered.groupby("date", as_index=False)["total_demand_units"].sum()
    st.plotly_chart(px.line(demand, x="date", y="total_demand_units",
                            title="Total daily demand across selected bank(s)"),
                    use_container_width=True)

elif page == "Inventory":
    title("Blood Inventory", "Operational demo inventory initialized from the latest supplied dataset snapshot.")
    inv = get_inventory(None if selected == "All blood banks" else selected)
    if not inv.empty:
        st.metric("Total recorded units", f"{int(inv.available_units.sum()):,}")
        st.dataframe(inv, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(inv, x="blood_group", y="available_units", color="bank_name", barmode="group", title="Available units by blood group and bank"), use_container_width=True)
        st.download_button("Download inventory CSV", inv.to_csv(index=False), "blood_inventory.csv", "text/csv")
    st.caption("This is a local demonstration ledger. It is separate from the historical training dataset and is not a live hospital inventory.")

elif page == "Record Donation":
    title("Record Donation", "Add a demo donation and update the selected bank's inventory ledger.")
    bank = bank_id_for_model()
    bank_row = banks[banks.blood_bank_id == bank].iloc[0]
    with st.form("record_donation_form"):
        group = st.selectbox("Blood group", BLOOD_GROUPS)
        units = st.number_input("Units collected", min_value=1, max_value=1000, value=1)
        donated_on = st.date_input("Donation date", value=date.today())
        notes = st.text_input("Notes (optional)")
        submitted = st.form_submit_button("Save donation", type="primary")
    if submitted:
        add_donation(bank, bank_row.blood_bank_name, group, units, donated_on.isoformat(), notes)
        st.success("Donation recorded and demo inventory updated.")
        st.rerun()
    st.subheader("Donation history")
    hist = get_donations(None if selected == "All blood banks" else selected)
    st.dataframe(hist, use_container_width=True, hide_index=True)

elif page == "Blood Requests":
    title("Blood Requests", "Record requests and units issued against the demo inventory ledger.")
    bank = bank_id_for_model()
    bank_row = banks[banks.blood_bank_id == bank].iloc[0]
    inv = get_inventory(bank)
    with st.form("record_blood_request_form"):
        group = st.selectbox("Blood group", BLOOD_GROUPS)
        requested = st.number_input("Units requested", min_value=1, max_value=1000, value=1)
        issued = st.number_input("Units issued", min_value=0, max_value=1000, value=0)
        request_on = st.date_input("Request date", value=date.today())
        submitted = st.form_submit_button("Save request", type="primary")
    if submitted:
        try:
            add_request(bank, bank_row.blood_bank_name, group, requested, issued, request_on.isoformat())
            st.success("Blood request saved and demo inventory updated.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    st.subheader("Request history")
    hist = get_requests(None if selected == "All blood banks" else selected)
    st.dataframe(hist, use_container_width=True, hide_index=True)

elif page == "Demand Forecasting":
    title("Demand Forecasting", "Forecast daily blood demand from historical bank and blood-group patterns.")
    col1,col2 = st.columns(2)
    bank = bank_id_for_model()
    group = col2.selectbox("Blood group", BLOOD_GROUPS)
    horizon = st.selectbox("Forecast horizon", [7,14,30], format_func=lambda x: f"{x} days")
    st.info("The forecasting module uses a separate Random Forest Regressor trained on lagged and rolling historical demand.")
    st.warning(f"The supplied dataset ends on {data['date'].max():%d %b %Y}. Forecast dates begin after that date; upload newer verified records before treating this as a current operational forecast.")
    if st.button("Train / refresh demand model"):
        with st.spinner("Training demand model..."):
            try:
                model, metrics = train_demand_model(data)
                st.cache_resource.clear()
                st.success(f"Demand model trained. MAE: {metrics['mae']:.2f} units | RMSE: {metrics['rmse']:.2f} units")
                st.json(metrics)
            except Exception as exc:
                st.error(str(exc))
    if st.button("Generate forecast", type="primary"):
        try:
            model = get_demand_model()
            forecast = forecast_demand(data, model, bank, group, horizon)
            st.session_state["forecast_result"] = forecast
            st.session_state["forecast_key"] = (bank, group, horizon)
        except Exception as exc:
            st.error(str(exc))
    key = (bank, group, horizon)
    forecast = st.session_state.get("forecast_result")
    if forecast is not None and st.session_state.get("forecast_key") == key:
        st.metric("Total predicted demand", f"{forecast['predicted_demand_units'].sum():,.1f} units")
        st.plotly_chart(px.line(forecast, x="date", y="predicted_demand_units",
                                markers=True, title=f"{group} demand forecast"),
                        use_container_width=True)
        st.dataframe(forecast, use_container_width=True, hide_index=True)
        st.download_button("Download forecast CSV", forecast.to_csv(index=False),
                           f"{bank}_{group}_demand_forecast.csv", "text/csv")

elif page == "Shortage Prediction":
    title("Stock Status Prediction", "Predict Adequate, Surplus, Low, Critical or Shortage using the notebook's Random Forest approach.")
    bank = bank_id_for_model()
    bank_row = banks[banks.blood_bank_id == bank].iloc[0]
    group = st.selectbox("Blood group", BLOOD_GROUPS)
    latest_row = data[(data.blood_bank_id == bank) & (data.blood_group == group)].sort_values("date").iloc[-1]
    st.caption(f"Latest historical record for this selection: {latest_row['date'].date()}")
    with st.form("prediction_form"):
        a,b,c = st.columns(3)
        opening = a.number_input("Opening stock (units)", min_value=0, max_value=10000,
                                 value=int(latest_row["opening_stock_units"]))
        donations = b.number_input("Expected donations (units)", min_value=0, max_value=10000,
                                   value=int(latest_row["donations_received_units"]))
        demand = c.number_input("Expected total demand (units)", min_value=0, max_value=10000,
                                value=int(latest_row["total_demand_units"]))
        d,e,f = st.columns(3)
        emergency = d.number_input("Emergency requests", min_value=0, value=int(latest_row["emergency_request_units"]))
        accident = e.number_input("Road-traffic accident requests", min_value=0, value=int(latest_row["road_traffic_accident_request_units"]))
        elective = f.number_input("Elective surgery requests", min_value=0, value=int(latest_row["elective_surgery_request_units"]))
        g,h,i = st.columns(3)
        thal = g.number_input("Thalassemia requests", min_value=0, value=int(latest_row["thalassemia_request_units"]))
        camp = h.selectbox("Donor camp held?", [0,1], index=int(latest_row["donor_camp_held"]),
                           format_func=lambda x: "Yes" if x else "No")
        forecast_date = i.date_input("Prediction date", value=date.today())
        j,k,l = st.columns(3)
        temp = j.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0,
                              value=float(latest_row["temperature_c"]))
        rain = k.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0,
                              value=float(latest_row["rainfall_mm"]))
        festival = l.selectbox("Festival period", sorted(data["festival_period"].dropna().unique()),
                               index=sorted(data["festival_period"].dropna().unique()).index(latest_row["festival_period"]))
        weekday = st.selectbox("Day of week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                               index=forecast_date.weekday())
        submitted = st.form_submit_button("Predict stock status", type="primary")
    if submitted:
        row = {
            "blood_bank_id": bank, "blood_bank_name": bank_row["blood_bank_name"],
            "city": bank_row["city"], "state": bank_row["state"], "blood_group": group,
            "opening_stock_units": opening, "donations_received_units": donations,
            "total_demand_units": demand, "emergency_request_units": emergency,
            "road_traffic_accident_request_units": accident,
            "elective_surgery_request_units": elective, "thalassemia_request_units": thal,
            "donor_camp_held": camp, "temperature_c": temp, "rainfall_mm": rain,
            "festival_period": festival, "day_of_week": weekday
        }
        try:
            model = get_classifier()
            status, probabilities = predict_stock_status(model, row)
            st.subheader("Prediction result")
            if status == "Shortage":
                st.error(f"Predicted stock status: {status}")
            elif status == "Critical":
                st.warning(f"Predicted stock status: {status}")
            elif status in ("Low",):
                st.warning(f"Predicted stock status: {status}")
            else:
                st.success(f"Predicted stock status: {status}")
            st.write("Model class probabilities")
            probs = pd.DataFrame({"Stock status": list(probabilities.keys()),
                                  "Probability": list(probabilities.values())})
            st.plotly_chart(px.bar(probs, x="Stock status", y="Probability",
                                   range_y=[0,1], title="Prediction confidence by class"),
                            use_container_width=True)
            st.caption("This is an educational model estimate. Confirm actual inventory and all operational decisions with authorized blood-bank staff.")
        except Exception as exc:
            st.error(str(exc))

elif page == "Blood Bank Network":
    title("Blood Bank Network", "Compare participating blood banks using the latest available dataset records.")
    snapshot = data[data.date == latest_date]
    network = snapshot.groupby(["blood_bank_name","city"], as_index=False).agg(
        opening_stock=("opening_stock_units","sum"),
        closing_stock=("closing_stock_units","sum"),
        daily_demand=("total_demand_units","sum"),
        unmet_demand=("unmet_demand_units","sum")
    )
    st.dataframe(network, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(network, x="blood_bank_name", y="closing_stock",
                           color="city", title="Latest closing stock by blood bank"),
                    use_container_width=True)
    st.plotly_chart(px.bar(network, x="blood_bank_name", y="unmet_demand",
                           color="city", title="Latest unmet demand by blood bank"),
                    use_container_width=True)

elif page == "Dataset Explorer":
    title("Dataset Explorer", "Explore the supplied 2024–2026 blood-bank dataset.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", f"{len(filtered):,}")
    c2.metric("Columns", len(filtered.columns))
    c3.metric("Date range", f"{filtered.date.min():%d %b %Y} – {filtered.date.max():%d %b %Y}")
    banks_filter = st.multiselect("Blood bank", sorted(filtered["blood_bank_name"].unique()))
    groups_filter = st.multiselect("Blood group", BLOOD_GROUPS)
    statuses_filter = st.multiselect("Stock status", STOCK_STATUSES)
    view = filtered.copy()
    if banks_filter: view = view[view.blood_bank_name.isin(banks_filter)]
    if groups_filter: view = view[view.blood_group.isin(groups_filter)]
    if statuses_filter: view = view[view.stock_status.isin(statuses_filter)]
    st.write(f"Showing {len(view):,} rows")
    st.dataframe(view.drop(columns=["month","year","day_of_month"]), use_container_width=True, hide_index=True)
    st.download_button("Download filtered dataset", view.to_csv(index=False),
                       "bloodbank_filtered_dataset.csv", "text/csv")

elif page == "Model Performance":
    title("Model Performance", "Evaluation of the trained Random Forest models.")
    if st.button("Train / refresh stock-status classifier"):
        with st.spinner("Training classifier..."):
            try:
                metrics = train_stock_status_model(data)
                st.cache_resource.clear()
                st.success("Classifier trained and saved.")
                st.json({"accuracy": metrics["accuracy"], "weighted_f1": metrics["weighted_f1"]})
            except Exception as exc:
                st.error(str(exc))
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        a,b,c = st.columns(3)
        a.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        b.metric("Weighted F1", f"{metrics['weighted_f1']:.2%}")
        c.metric("Test records", f"{metrics['test_rows']:,}")
        report = metrics["classification_report"]
        rows = []
        for label in STOCK_STATUSES:
            if label in report:
                rows.append({"Class": label, "Precision": report[label]["precision"],
                             "Recall": report[label]["recall"], "F1-score": report[label]["f1-score"],
                             "Support": report[label]["support"]})
        st.subheader("Per-class evaluation")
        st.dataframe(pd.DataFrame(rows).style.format({
            "Precision":"{:.3f}","Recall":"{:.3f}","F1-score":"{:.3f}"
        }), use_container_width=True, hide_index=True)
        cm = pd.DataFrame(metrics["confusion_matrix"], index=metrics["class_labels"],
                          columns=metrics["class_labels"])
        st.subheader("Confusion matrix")
        st.plotly_chart(px.imshow(cm, text_auto=True, aspect="auto",
                                  labels=dict(x="Predicted", y="Actual", color="Records"),
                                  title="Stock-status confusion matrix"),
                        use_container_width=True)
        st.caption("The test split is stratified and uses an 80/20 random split, matching the notebook's general evaluation setup. Leakage-prone post-outcome columns are excluded.")
    else:
        st.warning("Classifier metrics have not been generated yet. Click the training button above.")
    from utils.config import DEMAND_METRICS_PATH
    if DEMAND_METRICS_PATH.exists():
        demand_metrics = json.loads(DEMAND_METRICS_PATH.read_text(encoding="utf-8"))
        st.subheader("Demand forecasting evaluation")
        x,y,z = st.columns(3)
        x.metric("MAE", f"{demand_metrics['mae']:.2f} units")
        y.metric("RMSE", f"{demand_metrics['rmse']:.2f} units")
        z.metric("Test records", f"{demand_metrics['test_rows']:,}")
        st.caption("Demand metrics are calculated on a chronological holdout, with the latest 20% of dates reserved for evaluation.")

elif page == "Reports":
    title("Reports & Downloads", "Export the supplied data and useful summaries.")
    st.download_button("Download full dataset CSV", data.to_csv(index=False),
                       "blood_bank_demand_shortage_2024_2026.csv", "text/csv")
    summary = filtered.groupby(["blood_bank_name","blood_group","stock_status"], as_index=False).size()
    st.subheader("Stock status summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button("Download stock status summary", summary.to_csv(index=False),
                       "stock_status_summary.csv", "text/csv")

st.divider()
st.caption("Academic prototype only. Predictions are not validated for clinical, transfusion, inventory-release or emergency allocation decisions.")
