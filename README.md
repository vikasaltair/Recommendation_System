gcloud builds submit --tag gcr.io/ba-data-science/recommendation-system --project=ba-data-science

gcloud run deploy --image gcr.io/ba-data-science/recommendation-system --platform managed --project=ba-data-science --allow-unauthenticated
