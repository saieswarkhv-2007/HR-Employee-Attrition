from app import app, ensure_model_exists, predict_attrition


def test_predict_attrition_model_works():
    ensure_model_exists()

    payload = {
        "Age": 34,
        "BusinessTravel": "Travel_Rarely",
        "Department": "Research & Development",
        "DistanceFromHome": 6,
        "Education": 4,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "Gender": "Female",
        "HourlyRate": 42,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 4,
        "MaritalStatus": "Married",
        "MonthlyIncome": 5500,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 4,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
    }

    result = predict_attrition(payload)

    assert "risk_probability" in result
    assert "label" in result
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert result["label"] in {"Low risk", "High risk"}


def test_prediction_form_keeps_user_values_after_submit():
    client = app.test_client()
    response = client.post(
        "/prediction",
        data={
            "Age": 42,
            "Department": "Sales",
            "JobRole": "Sales Executive",
            "MonthlyIncome": 7200,
            "OverTime": "Yes",
            "DistanceFromHome": 18,
            "TotalWorkingYears": 11,
            "YearsAtCompany": 6,
            "JobSatisfaction": 2,
            "WorkLifeBalance": 2,
            "EnvironmentSatisfaction": 2,
            "BusinessTravel": "Travel_Frequently",
        },
    )

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'value="42"' in html
    assert 'value="7200"' in html
    assert 'value="18"' in html
    assert 'Sales Executive' in html
