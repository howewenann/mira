from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

classifier = TypeSafeClassifier(
    model="d1-3b",
    base_url="http://localhost:1234",
    api_key="xxx",
)


response = classifier.invoke(
    {
        "state": (
            "The deploy failed twice and customers are seeing 500s. "
            "Can someone look now?"
        ),
        "questions": {
            "urgent": Noul(instructions="Does this need attention right now?"),
            "team": Choice(
                instructions="Which team should pick this up?",
                criteria={
                    "infra": "Deploys, availability, and on-call incidents.",
                    "billing": "Payments, invoices, and subscriptions.",
                },
            ),
            "severity": Score(
                instructions="How severe is the impact?",
                criteria=["Cosmetic.", "Degraded for some users.", "Full outage."],
            ),
        },
    }
)

print("urgent: ", response.nouls["urgent"].noul)
print("team: ", response.choices["team"].choice, ", confidence: ", response.choices["team"].confidence)
print("severity: ", response.scores["severity"].score)