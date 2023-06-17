
FROM python:3.9.5-slim-buster
WORKDIR /app
COPY Dataset/Coffee.csv /app/Dataset/Coffee.csv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
CMD ["flask", "run", "--host=0.0.0.0"]

# docker tag ai-mini-app:latest anirudh30/ai-mini-app:latest
# docker push anirudh30/ai-mini-app:latest


# generate image using : docker build -t ai-mini-app .
# look at image using docker images
# to deploy create a yaml file and remember to give the same image name there
# kubectl delete deployment "deployment name" : to delete the deployment
# to deploy :  kubectl apply -f deployment.yaml
# minikube dashboard : to open the dashboard and we can see the pods
#kubectl get pods : to see the pods
# kubectl get services : to see the services
# kubectl get deployments : to see the deployments
# kubectl expose deployment ai-mini-app --type=NodePort --port=5000 : to expose the deployment
# minikube service ai-mini-app : to open the service

# minikube service "service name" in this example its ai-mini-app

# need to install docker desktop, kubernetes CLI and minikube