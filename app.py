import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, String, Float, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
import hashlib
from prophet import Prophet

st.set_page_config(page_title="Smart Finance Dashboard", layout="wide")

# ---------------- DATABASE ----------------
engine = create_engine("sqlite:///finance.db", echo=False)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# ---------------- TABLES ----------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)
    role = Column(String)


class Finance(Base):
    __tablename__ = "finance"
    id = Column(Integer, primary_key=True)
    username = Column(String)
    date = Column(String)
    category = Column(String)
    type = Column(String)
    amount = Column(Float)


Base.metadata.create_all(engine)

# ---------------- AUTO FIX OLD DATABASE ----------------
inspector = inspect(engine)
columns = [col['name'] for col in inspector.get_columns('users')]

if "role" not in columns:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'"))
        conn.commit()

# ---------------- PASSWORD HASH ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------- LOGIN ----------------
def login_page():

    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        user = session.query(User).filter_by(username=username).first()

        if user and user.password == hash_password(password):

            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user.role

            st.success("Login Successful")
            st.rerun()

        else:
            st.error("Invalid Username or Password")

# ---------------- REGISTER ----------------
def register_page():

    st.title("📝 Register")

    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")

    role = st.selectbox("Account Type", ["user", "admin"])

    if st.button("Register"):

        existing = session.query(User).filter_by(username=username).first()

        if existing:
            st.warning("Username already exists")

        else:

            new_user = User(
                username=username,
                password=hash_password(password),
                role=role
            )

            session.add(new_user)
            session.commit()

            st.success("Registration Successful! Please login.")

# ---------------- DATA UPDATE ----------------
def data_update():

    st.title("📝 Manual Finance Entry")

    date = st.date_input("Date")
    category = st.text_input("Category")
    type_option = st.selectbox("Type", ["Income", "Expense"])
    amount = st.number_input("Amount", min_value=0.0)

    if st.button("Add Record"):

        new_record = Finance(
            username=st.session_state.username,
            date=str(date),
            category=category,
            type=type_option,
            amount=amount
        )

        session.add(new_record)
        session.commit()

        st.success("Record Added Successfully")

    data = session.query(Finance).filter_by(
        username=st.session_state.username
    ).all()

    if data:

        df = pd.DataFrame([{
            "Date": d.date,
            "Category": d.category,
            "Type": d.type,
            "Amount": d.amount
        } for d in data])

        st.subheader("Saved Records")
        st.dataframe(df)

# ---------------- DASHBOARD ----------------
def dashboard():

    st.title("📊 Smart Finance Dashboard")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel File",
        type=["csv", "xlsx"]
    )

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.strip()

        required = ["Date", "Category", "Type", "Amount"]

        if not all(col in df.columns for col in required):
            st.error("File must contain Date, Category, Type, Amount")
            return

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        income = df[df["Type"] == "Income"]["Amount"].sum()
        expense = df[df["Type"] == "Expense"]["Amount"].sum()
        profit = income - expense

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Income", f"₹ {income:.2f}")
        c2.metric("Total Expense", f"₹ {expense:.2f}")
        c3.metric("Profit / Loss", f"₹ {profit:.2f}")

        # TREND GRAPH
        st.subheader("📈 Income vs Expense Trend")

        trend = df.groupby(["Date", "Type"])["Amount"].sum().unstack().fillna(0)

        fig, ax = plt.subplots()

        if "Income" in trend:
            ax.plot(trend.index, trend["Income"], marker="o", label="Income")

        if "Expense" in trend:
            ax.plot(trend.index, trend["Expense"], marker="o", label="Expense")

        ax.legend()
        ax.grid(True)

        plt.xticks(rotation=45)

        st.pyplot(fig)

        # CATEGORY GRAPH
        st.subheader("📊 Expense by Category")

        expense_df = df[df["Type"] == "Expense"]

        if not expense_df.empty:

            cat = expense_df.groupby("Category")["Amount"].sum()

            fig2, ax2 = plt.subplots()

            ax2.bar(cat.index, cat.values)

            plt.xticks(rotation=45)

            st.pyplot(fig2)

        # DONUT CHART
        st.subheader("🥧 Income vs Expense Distribution")

        pie = df.groupby("Type")["Amount"].sum()

        fig3, ax3 = plt.subplots()

        ax3.pie(pie, labels=pie.index, autopct="%1.1f%%")

        circle = plt.Circle((0, 0), 0.70, fc='white')
        fig3.gca().add_artist(circle)

        st.pyplot(fig3)

        # FORECAST
        st.subheader("🔮 Income Forecast")

        income_df = df[df["Type"] == "Income"].groupby("Date")["Amount"].sum().reset_index()

        income_df = income_df.rename(columns={"Date": "ds", "Amount": "y"})

        if len(income_df) > 2:
            model = Prophet(daily_seasonality=True)

            model.fit(income_df)

            future = model.make_future_dataframe(periods=180)

            forecast = model.predict(future)

            future_data = forecast[forecast["ds"] > income_df["ds"].max()]

            future_data["Month"] = future_data["ds"].dt.to_period("M")

            monthly_forecast = future_data.groupby("Month")["yhat"].sum().reset_index()

            monthly_forecast["Month"] = monthly_forecast["Month"].astype(str)

            # LINE GRAPH
            st.subheader("Forecast Line Graph")

            fig_line, ax_line = plt.subplots()

            ax_line.plot(monthly_forecast["Month"], monthly_forecast["yhat"], marker="o")

            ax_line.grid(True)

            plt.xticks(rotation=45)

            st.pyplot(fig_line)

            # BAR GRAPH
            st.subheader("Forecast Bar Chart")

            fig_bar, ax_bar = plt.subplots()

            ax_bar.bar(monthly_forecast["Month"], monthly_forecast["yhat"])

            ax_bar.grid(True)

            plt.xticks(rotation=45)

            st.pyplot(fig_bar)

            st.subheader("Forecast Table")
            st.dataframe(monthly_forecast)

        st.subheader("Uploaded Data Preview")
        st.dataframe(df)

        # ASK FEATURE
        st.subheader("🤖 Ask About Your Data")

        question = st.text_input("Ask something like:profit, loss")

        if question:
            question = question.lower()

            if "total income" in question:
                st.success(f"Total Income is ₹ {total_income:.2f}")

            elif "total expense" in question:
                st.success(f"Total Expense is ₹ {total_expense:.2f}")

            elif "profit" in question:
                if profit >= 0:
                    st.success(f"Total Profit is ₹ {profit:.2f}")
                else:
                    st.warning("Currently in Loss.")

            elif "loss" in question:
                if profit < 0:
                    st.error(f"Total Loss is ₹ {abs(profit):.2f}")
                else:
                    st.success("No Loss. You are in Profit!")

            elif "balance" in question:
                st.success(f"Current Balance is ₹ {profit:.2f}")

            elif "highest expense" in question:
                max_exp = expense_data.sort_values("Amount", ascending=False).head(1)
                if not max_exp.empty:
                    row = max_exp.iloc[0]
                    st.success(f"Highest Expense: ₹ {row['Amount']} in {row['Category']}")
                else:
                    st.info("No expense data found.")

            else:
                st.info("Try asking: total income, total expense, profit, loss, balance, highest expense")

    else:
        st.info("Upload a file to view dashboard.")
# ---------------- FORECASTING PAGE ----------------
def forecasting_page():

    st.title("📦 Future Prediction")

    uploaded_file = st.file_uploader(
        "Upload Updated Excel Dataset",
        type=["xlsx", "csv"]
    )

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        prophet_df = df.rename(columns={"Date": "ds", "Amount": "y"})

        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=180)

        forecast = model.predict(future)

        # EXISTING FORECAST GRAPH (UNCHANGED)
        fig = model.plot(forecast)
        st.pyplot(fig)

        # -------- CREATE MONTHLY FORECAST (THIS WAS MISSING) --------
        future_data = forecast[forecast["ds"] > prophet_df["ds"].max()]

        future_data["Month"] = future_data["ds"].dt.to_period("M")

        monthly_forecast = future_data.groupby("Month")["yhat"].sum().reset_index()

        monthly_forecast["Month"] = monthly_forecast["Month"].astype(str)

        # -------- LINE GRAPH --------
        st.subheader("Forecast Line Graph")

        fig_line, ax_line = plt.subplots()

        ax_line.plot(monthly_forecast["Month"], monthly_forecast["yhat"], marker="o")

        ax_line.grid(True)

        plt.xticks(rotation=45)

        st.pyplot(fig_line)

        # -------- BAR GRAPH --------
        st.subheader("Forecast Bar Chart")

        fig_bar, ax_bar = plt.subplots()

        ax_bar.bar(monthly_forecast["Month"], monthly_forecast["yhat"])

        ax_bar.grid(True)

        plt.xticks(rotation=45)

        st.pyplot(fig_bar)

        # -------- TABLE --------
        st.subheader("Forecast Table")
        st.dataframe(monthly_forecast)

        # -------- DATA PREVIEW --------
        st.subheader("Uploaded Data Preview")
        st.dataframe(df)

# ---------------- REPORT ----------------
def report_page():

    st.title("📑 Finance Report")

    data = session.query(Finance).filter_by(
        username=st.session_state.username
    ).all()

    if not data:
        st.info("No Data Available")
        return

    df = pd.DataFrame([{
        "Date": d.date,
        "Category": d.category,
        "Type": d.type,
        "Amount": d.amount
    } for d in data])

    df["Date"] = pd.to_datetime(df["Date"])

    monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()
    monthly.index = monthly.index.astype(str)

    fig, ax = plt.subplots()
    ax.bar(monthly.index, monthly.values)

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.subheader("Manual Data Table")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV Report",
        csv,
        "finance_report.csv",
        "text/csv"
    )

# ---------------- ADMIN PANEL ----------------
def admin_page():

    if st.session_state.role != "admin":
        st.error("Access Denied")
        return

    st.title("🛠 Admin Panel")

    total_users = session.query(User).count()

    income_count = session.query(Finance).filter_by(type="Income").count()
    expense_count = session.query(Finance).filter_by(type="Expense").count()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Users", total_users)
    c2.metric("Income Records", income_count)
    c3.metric("Expense Records", expense_count)

    users = session.query(User).all()

    user_df = pd.DataFrame([{
        "ID": u.id,
        "Username": u.username,
        "Role": u.role
    } for u in users])

    st.subheader("All Users")
    st.dataframe(user_df)

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

# BEFORE LOGIN
if not st.session_state.logged_in:

    page = st.sidebar.radio(
        "Account",
        ["Login", "Register"]
    )

    if page == "Login":
        login_page()
    else:
        register_page()

# AFTER LOGIN
else:

    if st.session_state.role == "admin":

        page = st.sidebar.radio(
            "Admin Menu",
            ["Dashboard", "Admin Panel", "Logout"]
        )

        if page == "Dashboard":
            dashboard()

        elif page == "Admin Panel":
            admin_page()

        elif page == "Logout":
            st.session_state.clear()
            st.rerun()

    else:

        page = st.sidebar.radio(
            "Menu",
            ["Dashboard", "Data Update", "Report", "Forecasting", "Logout"]
        )

        if page == "Dashboard":
            dashboard()

        elif page == "Data Update":
            data_update()

        elif page == "Report":
            report_page()

        elif page == "Forecasting":
            forecasting_page()

        elif page == "Logout":
            st.session_state.clear()
            st.rerun()
