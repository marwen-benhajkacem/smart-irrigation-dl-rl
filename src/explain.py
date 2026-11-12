"""Bloc 3 : traduire une décision en phrase pour l'agriculteur.
  (a) explain_template : règles déterministes, 100 % fiable, sert de BASE (aucune dépendance)
  (b) explain_llm      : reformulation par un LLM (API Anthropic) à partir des MÊMES faits -> pas d'invention
Le LLM ne décide RIEN : il reformule des faits chiffrés fournis par l'agent.
"""
import os


def facts(day, sw_now, sw_forecast, rain_today, irrig_mm, swfac=None):
    trend = sw_forecast[-1] - sw_now
    return {
        "jour": day, "humidite_actuelle": round(sw_now, 3), "humidite_prevue_j7": round(float(sw_forecast[-1]), 3),
        "tendance": "baisse" if trend < -0.01 else ("hausse" if trend > 0.01 else "stable"),
        "pluie_aujourdhui_mm": round(rain_today, 1), "decision_mm": irrig_mm,
        "stress": None if swfac is None else round(float(swfac), 2),
    }


def explain_template(f):
    if f["decision_mm"] == 0:
        if f["pluie_aujourdhui_mm"] > 2:
            why = f"il a plu ({f['pluie_aujourdhui_mm']} mm)"
        elif f["tendance"] != "baisse":
            why = "l'humidité du sol est suffisante et la prévision est stable ou en hausse"
        else:
            why = "le stock d'eau reste correct pour le moment"
        return f"Jour {f['jour']} : pas d'arrosage, car {why}."
    return (f"Jour {f['jour']} : arrosage de {f['decision_mm']:g} mm, car l'humidité du sol ({f['humidite_actuelle']}) "
            f"devrait évoluer en {f['tendance']} d'ici 7 jours (prévu : {f['humidite_prevue_j7']}) sans apport.")


def explain_llm(f, model="claude-haiku-5-5"):
    """Nécessite : pip install anthropic  et la variable d'environnement ANTHROPIC_API_KEY.
    Vérifiez le nom du modèle dans la documentation Anthropic avant usage."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = ("Tu es un conseiller agricole. Explique en 1 ou 2 phrases simples, en français, la décision d'irrigation "
              "suivante à un agriculteur. N'utilise QUE ces faits, n'invente aucun chiffre : " + str(f))
    r = client.messages.create(model=model, max_tokens=150, messages=[{"role": "user", "content": prompt}])
    return r.content[0].text


if __name__ == "__main__":
    import numpy as np
    f = facts(42, 0.14, np.array([0.13, 0.12, 0.12, 0.11, 0.11, 0.10, 0.10]), 0.0, 15.0, 0.82)
    print(explain_template(f))
