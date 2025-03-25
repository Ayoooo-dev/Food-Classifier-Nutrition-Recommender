import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.datasets import ImageFolder
import requests

# === Load class names from your dataset folder ===
from torchvision.datasets import ImageFolder
dataset = ImageFolder("D:/School!/CpEML/Project/images")
class_names = dataset.classes  # ← this must come first

# === Load model ===
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # Now this works
model.load_state_dict(torch.load("food_classifier.pth", map_location="cpu"))
model.eval()

# === Image transformation ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Streamlit UI ===
st.title("🍽️ Food Classifier + Nutrition Recommender")
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        st.success(f"🍕 Predicted: {predicted_class}")

        # === Optional: Fetch nutrition info
        app_id = "7848acbb"
        app_key = "46f366e81de8c0c7f5382ab71a4a8ac1"

        def get_nutrition_info(food_name):
            url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
            headers = {
                "x-app-id": app_id,
                "x-app-key": app_key,
                "Content-Type": "application/json"
            }
            data = {"query": food_name}
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()['foods'][0]
            return None

        food_info = get_nutrition_info(predicted_class)
        if food_info:
            st.write("### 🧾 Nutrition Info")
            st.write(f"Calories: {food_info['nf_calories']} kcal")
            st.write(f"Protein: {food_info['nf_protein']} g")
            st.write(f"Fat: {food_info['nf_total_fat']} g")
            st.write(f"Carbs: {food_info['nf_total_carbohydrate']} g")
            st.write(f"Sugar: {food_info['nf_sugars']} g")

            st.write("### 📌 Recommendations")
            if food_info['nf_protein'] < 10:
                st.warning("Add more protein (e.g. chicken, beans, tofu).")
            if food_info['nf_total_fat'] > 20:
                st.warning("Reduce fat intake.")
            if food_info['nf_sugars'] > 15:
                st.warning("Cut back on sugar.")
            if (food_info['nf_protein'] >= 10 and 
                food_info['nf_total_fat'] <= 20 and 
                food_info['nf_sugars'] <= 15):
                st.success("Looks balanced! ✅")
