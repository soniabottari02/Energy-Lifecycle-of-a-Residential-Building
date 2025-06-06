import streamlit as st
import pandas as pd
import joblib
import os
import json
from model_utils import train_model_from_excel
import numpy as np
import matplotlib.pyplot as plt

# Configurazione pagina
st.set_page_config(page_title="Predizione EPT", layout="centered")
st.title("Previsione del valore EPT con XGBoost")

# STEP 1: Carica Excel e allena il modello
with st.expander("1) Carica file Excel e allena il modello"):
    excel_file = st.file_uploader("Scegli un file Excel (.xlsx)", type=["xlsx"])
    if st.button("Allena modello"):
        if excel_file is None:
            st.warning("Devi prima caricare un file Excel.")
        else:
            with open("dati.xlsx", "wb") as f:
                f.write(excel_file.read())
            try:
                model, encoder, score, feature_names, cat_cols = train_model_from_excel("dati.xlsx")
                st.success(f"Modello addestrato! Precisione (R²): {score:.3f}")
                with open("features.json", "w") as jf:
                    json.dump({"features": feature_names, "cat_cols": cat_cols}, jf)
                st.info("Lista di feature salvata in features.json")
            except Exception as e:
                st.error(f"Errore durante l'addestramento: {e}")

# STEP 2: Form per la previsione
if os.path.exists("xgb_ept_model.pkl"):
    st.markdown("---")
    st.header("2) Fai una previsione EPT")

    # Carico feature e cat_cols da file
    try:
        with open("features.json", "r") as jf:
            data = json.load(jf)
            feature_names = data.get("features", [])
            cat_cols = data.get("cat_cols", [])
    except Exception:
        st.error("Impossibile leggere features.json.")
        st.stop()

    if not feature_names:
        st.error("Non ci sono feature disponibili. Addestra prima il modello.")
        st.stop()

    # Costruisco il form
    with st.form("input_form"):
        user_input = {}
        for feat in feature_names:
            if feat in cat_cols:
                user_input[feat] = st.text_input(f"{feat} (testuale)")
            else:
                # Usa text_input per numeri per gestire virgole come decimali
                user_input[feat] = st.text_input(f"{feat} (numerico, usa '.' o ',')")

        submitted = st.form_submit_button("Calcola EPT")

    if submitted:
        # Converto input numerici sostituendo ',' con '.' e trasformo in float
        for k, v in user_input.items():
            if k not in cat_cols:
                try:
                    user_input[k] = float(v.replace(',', '.'))
                except Exception:
                    st.error(f"Valore non valido per {k}: {v}")
                    st.stop()

        try:
            # Carico modello e encoder
            model = joblib.load("xgb_ept_model.pkl")
            encoder = joblib.load("ept_encoder.pkl") if os.path.exists("ept_encoder.pkl") else None

            # Preparo DataFrame di input
            df_input = pd.DataFrame([user_input])

            # Encoding variabili categoriali (se presenti)
            if encoder is not None and cat_cols:
                df_cat = pd.DataFrame(
                    encoder.transform(df_input[cat_cols]).toarray(),
                    columns=encoder.get_feature_names_out(cat_cols)
                )
                df_input = pd.concat([df_input.drop(columns=cat_cols), df_cat], axis=1)

            # Aggiungo colonne mancanti e riordino secondo feature_names
            for col in feature_names:
                if col not in df_input.columns:
                    df_input[col] = 0.0
            df_input = df_input[feature_names]

            # DEBUG: mostra input finale
            st.write("Input al modello:")
            st.write(df_input)

            # Previsione
            pred = model.predict(df_input)[0]
            st.success(f"Valore EPT previsto: {pred:.4f}")

            # Mappa codici → tipologie di combustibile
            mappa_combustibili = {
                1: "Biomassa",
                2: "Energia elettrica",
                3: "Gas naturale",
                4: "Gasolio",
                5: "GPL",
                6: "Olio combustibile",
                7: "RSU",
                8: "Teleriscaldamento",
                9: "Biomassa|Energia elettrica",
                10: "Biomassa|RSU",
                11: "Energia elettrica|RSU",
                12: "Gas naturale|Biomassa",
                13: "Gas naturale|Energia elettrica",
                14: "Gas naturale|Gasolio",
                15: "Gas naturale|GPL",
                16: "Gas naturale|RSU",
                17: "Gasolio|Biomassa",
                18: "Gasolio|Energia elettrica",
                19: "GPL|Biomassa",
                20: "GPL|Energia elettrica",
                21: "GPL|Gasolio",
                22: "Olio combustibile|Biomassa",
                23: "Olio combustibile|Biomassa|Energia elettrica",
                24: "Olio combustibile|Energia elettrica"
            }

            # Dizionario costi medi (€/unità EPT)
            costi_energia = {
                "Biomassa": 40.0,               # €/MWh
                "Energia elettrica": 0.16053,   # €/kWh
                "Gas naturale": 0.402,          # €/Smc
                "Gasolio": 1.43035,             # €/litro
                "GPL": 0.8,                     # €/litro (media)
                "Olio combustibile": 1.07,      # €/litro (media)
                "RSU": 15.0,                    
                "Teleriscaldamento": 46.0       # €/MWh
            }

            # Recupero codice numerico dal form
            try:
                codice_comb = int(user_input.get("TIPOLOGIA_COMBUSTIBILE", -1))
                tipo_energia = mappa_combustibili.get(codice_comb)
            except Exception:
                st.error("⚠️ Errore nella lettura della tipologia di combustibile.")
                st.stop()

            if not tipo_energia:
                st.warning(f"⚠️ Codice combustibile non valido: {codice_comb}")
                st.stop()

            # calcola costo medio (per combinazioni fa la media aritmetica,
            parti = [p.strip() for p in tipo_energia.split("|")]
            costi_presenti = [costi_energia[p] for p in parti if p in costi_energia]

            if not costi_presenti:
                st.warning("Costo non disponibile per la combinazione selezionata.")
                st.stop()

            costo_medio = np.mean(costi_presenti)

            # Tasso inflazione fisso (2%)
            tasso_inflazione = 0.02

            # Calcola costi nominali per 15 anni
            anni = np.arange(1, 16)
            costo_base = pred * costo_medio
            costi_futuri = costo_base * (1 + tasso_inflazione) ** anni




            # Calcolo dei costi attualizzati
            st.markdown("### Piano finanziario: costi attualizzati")

            # Tasso di attualizzazione (es. 3% annuo)
            tasso_attualizzazione = 0.03

            costi_attualizzati = [
                (pred * costo_medio) * ((1 + tasso_inflazione) ** i) / ((1 + tasso_attualizzazione) ** i)
                for i in anni
            ]
            totale_attualizzato = sum(costi_attualizzati)

            df_attualizzati = pd.DataFrame({
                "Anno": anni,
                "Costo attualizzato (€)": np.round(costi_attualizzati, 2)
            })

            st.dataframe(df_attualizzati, use_container_width=True)
            st.success(f"**Totale costi attualizzati (15 anni): € {totale_attualizzato:,.2f}**")

            # GRAFICO dei costi attualizzati
            fig2, ax2 = plt.subplots()
            ax2.plot(anni, costi_attualizzati, marker='o', color='blue')
            ax2.set_title("Costi attualizzati su 15 anni")
            ax2.set_xlabel("Anno")
            ax2.set_ylabel("Costo attualizzato (€)")
            ax2.grid(True)
            st.pyplot(fig2)



        except Exception as e:
            st.error(f"❌ Errore durante la previsione: {e}")
