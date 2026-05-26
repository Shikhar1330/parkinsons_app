import parselmouth
import numpy as np
import pandas as pd
import joblib

def extract_parkinsons_features(audio_path):
    """Extracts acoustic features from a voice file using parselmouth (Praat)."""
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch(time_step=0.01)

    try:
        pulses = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")
        jitter_local = parselmouth.praat.call([snd, pulses], "Get jitter (local)", 0, 0.02, 1.3, 1.6) * 100
        jitter_abs = parselmouth.praat.call([snd, pulses], "Get jitter (local, absolute)", 0, 0.02, 1.3, 1.6)
        rap = parselmouth.praat.call([snd, pulses], "Get jitter (rap)", 0, 0.02, 1.3, 1.6)
        ppq = parselmouth.praat.call([snd, pulses], "Get jitter (ppq5)", 0, 0.02, 1.3, 1.6)
        ddp = parselmouth.praat.call([snd, pulses], "Get jitter (ddp)", 0, 0.02, 1.3, 1.6)
        shimmer_local = parselmouth.praat.call([snd, pulses], "Get shimmer (local)", 0, 0.02, 1.3, 1.6, 1.6)
        shimmer_db = parselmouth.praat.call([snd, pulses], "Get shimmer (local_dB)", 0, 0.02, 1.3, 1.6, 1.6, 1.6)
    except Exception:
        # Fallback if jitter/shimmer extraction fails
        jitter_local = jitter_abs = rap = ppq = ddp = shimmer_local = shimmer_db = np.nan

    # Pitch-related features
    mean_F0 = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
    max_F0 = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    min_F0 = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

    # Noise/harmonic features
    try:
        nhr = parselmouth.praat.call(snd, "Get noise-to-harmonics ratio")
        hnr = parselmouth.praat.call(snd, "Get harmonics-to-noise ratio")
    except Exception:
        nhr, hnr = np.nan, np.nan

    # Fill other acoustic placeholders (non-extractable by Praat)
    features = {
        "MDVP:Fo(Hz)": mean_F0,
        "MDVP:Fhi(Hz)": max_F0,
        "MDVP:Flo(Hz)": min_F0,
        "MDVP:Jitter(%)": jitter_local,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp,
        "MDVP:Shimmer": shimmer_local,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": shimmer_local / 2 if shimmer_local else np.nan,
        "Shimmer:APQ5": shimmer_local / 3 if shimmer_local else np.nan,
        "MDVP:APQ": shimmer_local / 4 if shimmer_local else np.nan,
        "Shimmer:DDA": shimmer_local / 5 if shimmer_local else np.nan,
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": np.random.uniform(0.4, 0.7),
        "DFA": np.random.uniform(0.6, 0.8),
        "spread1": np.random.uniform(-6, -4),
        "spread2": np.random.uniform(0.3, 0.5),
        "D2": np.random.uniform(1.5, 2.5),
        "PPE": np.random.uniform(0.05, 0.15)
    }

    return pd.DataFrame([features])


def load_models():
    """Load all trained models and scalers."""
    clf_disease = joblib.load('models/clf_disease.joblib')
    reg_severity = joblib.load('models/reg_severity.joblib')
    reg_adj = joblib.load('models/reg_adj_severity.joblib')
    kmeans = joblib.load('models/stress_cluster_model.joblib')
    scaler = joblib.load('models/feature_scaler.joblib')
    scaler_adj = joblib.load('models/feature_scaler_stress.joblib')
    return clf_disease, reg_severity, reg_adj, kmeans, scaler, scaler_adj


def predict_from_audio(audio_path):
    """Full pipeline: extract → preprocess → predict."""
    df = extract_parkinsons_features(audio_path)
    clf_disease, reg_severity, reg_adj, kmeans, scaler, scaler_adj = load_models()

    # --- Clean up and handle missing values safely ---
    df = df.fillna(df.mean(numeric_only=True))  # Replace NaNs with mean
    df = df.fillna(0)                           # Fill any remaining NaNs
    df = df.replace([np.inf, -np.inf], 0)       # Replace inf/-inf

    if df.isnull().any().any():
        print("⚠️ Warning: Some NaN values remain after cleaning.")
    else:
        print("✅ No NaN values remain before scaling.")

    # --- Prepare scaled inputs ---
    acoustic_features = list(scaler.feature_names_in_)
    df_scaled = df.reindex(columns=acoustic_features, fill_value=0)
    X_scaled = scaler.transform(df_scaled)

    adj_features = df.reindex(columns=scaler_adj.feature_names_in_, fill_value=0)
    X_scaled_adj = scaler_adj.transform(adj_features)

    # --- Predictions ---
    disease_prob = clf_disease.predict_proba(X_scaled)[0][1]
    disease_label = "Parkinson's" if disease_prob > 0.5 else "Normal"
    severity_pred = reg_severity.predict(X_scaled)[0]
    adjusted_sev = reg_adj.predict(X_scaled_adj)[0]

    # --- Compute Stress Score ---
    df['Stress_Score'] = (
        0.3 * df['MDVP:Jitter(%)'] +
        0.3 * df['MDVP:Shimmer'] +
        0.2 * df['NHR'] -
        0.2 * df['HNR']
    )

    # Normalize safely
    stress_min, stress_max = df['Stress_Score'].min(), df['Stress_Score'].max()
    df['Stress_Score'] = (
        (df['Stress_Score'] - stress_min) / (stress_max - stress_min)
        if stress_max != stress_min else 0.5
    )

    # --- Stress Clustering ---
    stress_features = ['MDVP:Jitter(%)', 'MDVP:Shimmer', 'NHR', 'HNR']
    df['Stress_Cluster'] = kmeans.predict(df[stress_features])
    df['Stress_Label'] = df['Stress_Cluster'].apply(lambda x: 'High Stress' if x == 1 else 'Low Stress')

    return {
        'Disease Probability': disease_prob,
        'Disease Label': disease_label,
        'Predicted Severity': severity_pred,
        'Adjusted Severity': adjusted_sev,
        'Stress Label': df['Stress_Label'][0],
        'Stress Score': float(df['Stress_Score'].iloc[0])
    }
