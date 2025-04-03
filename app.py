import gradio as gr
import google.generativeai as genai
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from fpdf import FPDF
import os

# Set up Gemini API Key
genai.configure(api_key="AIzaSyCB8QoP1DPjMB8yBOSPyZnM7BG_l-F315A")

# Define preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names
class_names = ["Normal/Mild", "Moderate", "Severe"]

# Define the SpineClassifier Model
class SpineClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(SpineClassifier, self).__init__()
        self.features = models.efficientnet_b0(pretrained=True)
        self.features.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x):
        return self.features(x)

# Register hooks to capture activations and gradients
def register_hooks(model):
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def save_activation(act):
        activations.append(act)

    # Get the last convolutional layer and register the hooks
    # Correct way to access the last convolutional layer in EfficientNet
    last_conv_layer = list(model.features.features.children())[-1]
    last_conv_layer.register_forward_hook(lambda self, input, output: save_activation(output))
    last_conv_layer.register_backward_hook(lambda self, grad_input, grad_output: save_gradient(grad_output[0]))

    return activations, gradients

# Disease condition descriptions
disease_info = {
    "Normal/Mild": ("No significant issue", "Minimal or no significant narrowing or degeneration."),
    "Moderate": ("Moderate spinal issues", "Moderate narrowing or degeneration, potentially causing mild symptoms."),
    "Severe": ("Severe spinal issues", "Significant narrowing or degeneration, likely leading to more pronounced symptoms."),
}

# Initialize the model and hooks
model = SpineClassifier(num_classes=3)
model.eval()

# Initialize hook variables
activations = []
gradients = []
activations, gradients = register_hooks(model)

# Grad-CAM Heatmap Generation
def generate_heatmap(image):
    # Clear previous activations and gradients
    activations.clear()
    gradients.clear()
    
    # Apply transformations to the image and ensure it requires gradients
    image_tensor = transform(image).unsqueeze(0)
    image_tensor.requires_grad = True
    
    # Forward pass
    output = model(image_tensor)
    
    # Get the class with the highest score
    pred_class = output.argmax().item()

    # Zero all the gradients
    model.zero_grad()
    
    # Create a one-hot vector for the predicted class
    one_hot = torch.zeros_like(output)
    one_hot[0][pred_class] = 1

    # Backward pass with the one-hot vector as the gradient
    output.backward(gradient=one_hot)
    
    # Get the activations and gradients from the hooks
    activation = activations[0]
    gradient = gradients[0]
    
    # Compute the Grad-CAM heatmap
    weights = torch.mean(gradient, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activation, dim=1).squeeze().detach().numpy()
    
    # Normalize the heatmap
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize

    # Convert image to numpy array for cv2 processing
    img_np = np.array(image.resize((224, 224)))
    
    # Check if image is grayscale and convert to RGB if needed
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Convert RGB to BGR for OpenCV
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Apply colormap to create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on the original image
    superimposed = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    
    # Convert back to RGB for PIL
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    
    # Create PIL image from numpy array
    heatmap_img = Image.fromarray(superimposed)
    
    # Save the heatmap image
    plt.imsave("heatmap.png", superimposed)
    
    return heatmap_img

# Prediction function
def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = class_names[output.argmax().item()]
    
    condition, detail = disease_info[predicted_class]
    heatmap = generate_heatmap(image)
    
    return predicted_class, condition, detail, heatmap

# Generate and Download PDF Report
def generate_report(image, disease, condition, detail):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, "Spine Disease Detection Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Predicted Disease: {disease}", ln=True)
    pdf.cell(200, 10, f"Condition: {condition}", ln=True)
    pdf.multi_cell(0, 10, f"Details: {detail}")
    pdf.ln(10)
    
    # Ensure heatmap.png exists
    if not os.path.exists("heatmap.png"):
        generate_heatmap(image)
        
    pdf.image("heatmap.png", x=40, w=130)
    pdf.ln(10)
    
    # Save to a specific directory to avoid IDM interception
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "spine_report.pdf")
    pdf.output(report_path)
    
    return report_path

# Use Gemini API for more advanced chatbot responses
def improved_chatbot_response(query):
    # Define a comprehensive set of responses for specific questions
    fixed_responses = {
        "what is normal/mild spinal disease": "Normal/Mild refers to minimal or no significant narrowing or degeneration of the spine.",
        "what is moderate spinal disease": "Moderate spinal disease involves moderate narrowing or degeneration of the spine, potentially causing mild symptoms.",
        "what is severe spinal disease": "Severe spinal disease involves significant narrowing or degeneration of the spine, which can lead to more pronounced symptoms.",
        "what are the symptoms of spinal disease": "Symptoms of spinal disease may include pain, stiffness, weakness, or limited movement in the affected area.",
        "how is spinal disease treated": "Spinal disease treatments may include physical therapy, medications, injections, or surgery depending on the severity.",
    }
    
    # Normalize the query for case-insensitive matching
    normalized_query = query.lower().strip()
    
    # Check for exact matches in our fixed responses first
    for key, response in fixed_responses.items():
        if key in normalized_query:
            return response
    
    # For other spine-related queries, use the Gemini API
    try:
        if any(keyword in normalized_query for keyword in ["spine", "spinal", "back", "vertebra", "disc", "disk", "nerve", "pain", "posture", "mri", "x-ray"]):
            # Use Gemini API for spine-related questions
            model = genai.GenerativeModel('gemini-1.5-pro')
            prompt = f"Answer this question about spine health concisely (max 3 sentences): {query}"
            response = model.generate_content(prompt)
            return response.text
        else:
            # For non-spine related queries
            return "I'm sorry, I can only answer questions related to spinal diseases and spine health."
    except Exception as e:
        # Fallback in case of API error
        return f"I'm having trouble connecting to my knowledge base. Please try asking about specific spine conditions like Normal/Mild, Moderate, or Severe spinal disease."

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("# 🏥 Spine Disease Detector with AI Chat Support")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Spine MRI/X-ray")
    
    classify_btn = gr.Button("🔍 Predict Disease")
    
    with gr.Row():
        disease_output = gr.Textbox(label="Predicted Class", interactive=False)
        condition_output = gr.Textbox(label="Condition", interactive=False)
    
    detail_output = gr.Textbox(label="Detail", interactive=False)
    heatmap_output = gr.Image(label="Heatmap Analysis")
    
    download_btn = gr.Button("📥 Download Report")
    report_output = gr.File(label="Download PDF Report")
    
    gr.Markdown("## 💬 Ask AI About Spine Health")
    
    # Chatbot UI with explicit submit button
    with gr.Row():
        query_input = gr.Textbox(label="Your Question", placeholder="Ask about spine health or conditions...")
    
    submit_btn = gr.Button("🔍 Ask AI")
    response_output = gr.Textbox(label="AI Response", interactive=False)
    
    # Define actions
    classify_btn.click(predict, inputs=[image_input], outputs=[disease_output, condition_output, detail_output, heatmap_output])
    download_btn.click(generate_report, inputs=[image_input, disease_output, condition_output, detail_output], outputs=[report_output])
    
    # Add both submit button click and Enter key functionality for the chatbot
    submit_btn.click(improved_chatbot_response, inputs=[query_input], outputs=[response_output])
    query_input.submit(improved_chatbot_response, inputs=[query_input], outputs=[response_output])

# Run the Gradio App
if __name__ == "__main__":
    interface.launch(share=True)