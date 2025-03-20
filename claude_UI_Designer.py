import json
import requests

# Set up API key
ANTHROPIC_API_KEY = ""

# Define Claude API endpoint
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Function to send a request to Claude 3
def get_claude_response(prompt, api_key):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["content"]  # No need for json.loads()
    else:
        print("❌ API Error:", response.text)
        return None

# Step 1: Get user input for design request
design_request = input("Describe the UI design you want: ")

# Step 2: Generate multiple design options
prompt = f"""
You are an expert UI/UX designer. Generate 3 different UI design concepts based on the user's request: "{design_request}". 
For each concept, provide:
1. **Title** (Short name of the design)
2. **Layout Structure** (Grid, Sidebar, Cards, etc.)
3. **Color Scheme** (Dark, Light, Accent Colors)
4. **Component Details** (Buttons, Inputs, Tables, etc.)
5. **Figma-style description** (How the UI elements are arranged)

Output the designs as a **Python list of dictionaries** inside JSON block.
"""

response = get_claude_response(prompt, ANTHROPIC_API_KEY)

if response:
    try:
        # Debug: Print the response structure
        print("\n🛠 Debug - Raw Response:\n", response)

        # Extract the text content from the response list
        if isinstance(response, list) and response:
            response_text = response[0].get("text", "")
        else:
            response_text = response  # If it's already a string

        # Find the JSON block inside the response text
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        json_block = response_text[json_start:json_end]

        # Convert JSON string to Python list
        design_options = json.loads(json_block)

        # Debug: Print structured response
        print("\n📜 Parsed UI Designs:", json.dumps(design_options, indent=4))

        # Step 3: Display design options
        print("\n🎨 Available UI Designs:\n")
        for idx, design in enumerate(design_options, 1):
            print(f"{idx}. {design.get('Title', 'Untitled Design')}")
            print(f"   📐 Layout: {design.get('Layout Structure', 'Not provided')}")
            print(f"   🎨 Colors: {design.get('Color Scheme', 'Not provided')}")
            print(f"   🔹 Components: {design.get('Component Details', 'Not provided')}\n")

        # Step 4: Let the user pick a design
        choice = int(input("Enter the number of the design you want: ")) - 1

        if 0 <= choice < len(design_options):
            selected_design = design_options[choice]

            # Ensure layout and colors are present
            layout = selected_design.get('Layout Structure', 'Unknown')
            colors = selected_design.get('Color Scheme', 'Unknown')

            if layout == "Unknown" or colors == "Unknown":
                print("❌ Error: The selected design is missing layout or color details. Please choose a different design.")
            else:
                # Step 5: Generate HTML & CSS for the chosen design
                prompt = f"""
                Based on this UI design:
                - **Title**: {selected_design.get('Title', 'Untitled')}
                - **Layout**: {layout}
                - **Colors**: {colors}
                - **Components**: {selected_design.get('Component Details', 'Not provided')}

                🔹 Generate a **fully responsive** HTML & CSS code snippet that follows this design.
                🔹 Ensure the sidebar is styled properly and card widgets are responsive.
                🔹 The color scheme should match the provided details.
                🔹 Use modern HTML5 & CSS3 practices (e.g., Flexbox, Grid).
                🔹 Provide only the complete and functional HTML & CSS code, without explanations.
                """

                ui_code = get_claude_response(prompt, ANTHROPIC_API_KEY)

                if ui_code:
                    # Save the generated code
                    output_file = "generated_ui.html"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(ui_code if isinstance(ui_code, str) else json.dumps(ui_code, indent=4))

                    print(f"\n✅ UI code saved to {output_file}. Open it in a browser to view.")

        else:
            print("❌ Invalid selection. Please restart and choose a valid option.")

    except json.JSONDecodeError:
        print("❌ Error: Failed to parse the JSON response from Claude 3.")
