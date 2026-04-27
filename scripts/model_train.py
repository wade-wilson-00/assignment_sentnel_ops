import os
import joblib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.model_selection import train_test_split 
from llm_reasoning import LLM_Reasoning

class AnamolyDetection:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=110,
            max_depth=5,
            min_samples_leaf=5,
            random_state=48,
            min_samples_split=2,
        )
        self.llm_response = LLM_Reasoning()

    def load_data(self):
        self.df = pd.read_csv("data/synthetic_metrics.csv")
        self.df.head()
        return self.df
    
    def preprocessing(self):
        self.X = self.df.drop(["is_anomalous","resource_id","anomaly_type"], axis=1) 
        self.y = self.df["anomaly_type"]
        print(f"Shape of X:{self.X.shape}")
        print(f"Shape of Y:{self.y.shape}")

        self.X["internet_facing"] = self.X["internet_facing"].astype(int)
        self.X["identity_attached"] = self.X["identity_attached"].astype(int)

        #Engineered Features
        #["cpu_spike_ratio"] = ["cpu_p95"] / (["cpu_avg"] + 1)
        #["resource_saturation"] = ["cpu_avg"] + (["memory_avg"] / 2)
        #["load_efficiency"] = ["network_pct"] / (["cpu_avg"] + 1)

        return self.X, self.y
    
    def data_split(self):
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y, test_size=0.2, random_state=40)
        print(f"Shape of X_train - {self.X_train.shape}")
        print(f"Shape of X_test - {self.X_test.shape}")

        print(f"Shape of y_train - {self.y_train.shape}")
        print(f"Shape of y_test - {self.y_test.shape}")

        return self.X_train,self.X_test,self.y_train,self.y_test
    
    def model_train(self):
        self.model.fit(self.X_train, self.y_train)
        print("Model is Trained !")

        return self.model
    
    def get_prediction(self):
        self.y_pred = self.model.predict(self.X_test)
        self.confidence_score = self.model.predict_proba(self.X_test)
        
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        # Use average='weighted' for multiclass classification
        self.precision = precision_score(self.y_test, self.y_pred, average='weighted')
        self.recall = recall_score(self.y_test, self.y_pred, average='weighted')

        print(f"\n--- Model Performance ---")
        print(f"Accuracy : {self.accuracy * 100:.2f}%")
        print(f"Precision: {self.precision * 100:.2f}%")
        print(f"Recall   : {self.recall * 100:.2f}%")

        return self.y_pred, self.accuracy, self.precision, self.recall
    
    def context_package(self, i):

        idx = self.X_test.index[i]
        sample = self.df.loc[idx]
        prediction = self.y_pred[i]
        confidence_score = np.max(self.confidence_score[i]) 

        context = f"""
        Resource Metrics:
        resource_id:{sample["resource_id"]}
        cpu_avg:{sample["cpu_avg"]}%
        network_pct:{sample["network_pct"]}%
        internet_facing:{sample["internet_facing"]}

        ML Predictions :
        -prediction:{prediction}
        -confidence_score:{confidence_score}
        -is_anomalous:{sample["is_anomalous"]}
        """
        return context
    def get_response(self, context): 

        prompt = f"""
        You're a Smart resource analyst and an anomaly detection specialist, 
        based on the given metrics{context} by the user and based on the predictions 
        and confidence_score inside the context, you have to give a detailed resoning
        in strictly this JSON Format:
        {{
            "resource_id": "original_id",
            "is_anomalous": true/false,
            "anomaly_type": "the_predicted_label",
            "reason": "a short one line explanation",
            "suggested_action": "what to do next",
            "confidence": 0.00,
            "security_note": "only if relevant"
        }}
        """
        response = self.llm_response.llm_brain(
            content=prompt
        )

        # Strip preamble text to keep only the JSON object
        if "{" in response:
            response = response[response.find('{'):response.rfind('}')+1]
        return response

if __name__ == "__main__":
    detector = AnamolyDetection()
    detector.load_data()
    detector.preprocessing()
    detector.data_split()
    detector.model_train()
    detector.get_prediction()
    
    print("\n" + "="*50)
    print("🚀 GENERATING 5 SAMPLE ANALYSES")
    print("="*50)

    for i in range(5):
        ctx = detector.context_package(i)
        result = detector.get_response(ctx)
        print(f"\n--- [SAMPLE {i+1}] ---")
        print(result)
