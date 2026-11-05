import pandas as pd

FEATURE_COLUMNS = [
    "CustomerAge",
    "TenureMonths",
    "MonthlyCharges",
    "SupportTickets",
    "UsageHours",
    "ContractType",
]

FIELD_LABELS = {
    "CustomerAge": "Customer age (years)",
    "TenureMonths": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "SupportTickets": "Support tickets",
    "UsageHours": "Usage hours",
    "ContractType": "Contract type",
}

INT_COLUMNS = {"CustomerAge", "TenureMonths", "SupportTickets"}

NUMERIC_BOUNDS = {
    "CustomerAge": (0, 120),
    "TenureMonths": (0, 120),
    "MonthlyCharges": (0.0, 1000.0),
    "SupportTickets": (0, 100),
    "UsageHours": (0.0, 1000.0),
}

CATEGORICAL_OPTIONS = {
    "ContractType": ["Monthly", "Yearly"],
}

FORM_SECTIONS = [
    ("Customer", ["CustomerAge", "TenureMonths"]),
    ("Usage", ["MonthlyCharges", "SupportTickets", "UsageHours"]),
    ("Contract", ["ContractType"]),
]


def validate_batch_frame(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    unexpected = [str(column) for column in df.columns if column not in FEATURE_COLUMNS]

    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}.")
    if unexpected:
        warnings.append(
            f"Unexpected columns: {', '.join(unexpected)}. They will not be sent to the model."
        )
    if df.empty:
        errors.append("The file contains no customer rows.")
        return errors, warnings

    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue
        if df[column].isna().any():
            errors.append(f"{column} has empty values.")
            continue
        if column in CATEGORICAL_OPTIONS:
            invalid = sorted(set(df[column].astype(str)) - set(CATEGORICAL_OPTIONS[column]))
            if invalid:
                errors.append(f"{column} has unsupported values: {', '.join(invalid)}.")
            continue
        numbers = pd.to_numeric(df[column], errors="coerce")
        if numbers.isna().any():
            errors.append(f"{column} must contain only numbers.")
            continue
        low, high = NUMERIC_BOUNDS[column]
        if ((numbers < low) | (numbers > high)).any():
            errors.append(f"{column} must be between {low} and {high}.")

    return errors, warnings


def batch_template_csv() -> bytes:
    return pd.DataFrame(columns=FEATURE_COLUMNS).to_csv(index=False).encode("utf-8")
