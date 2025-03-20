import json
import ollama  # Using Ollama for local Mistral inference
import re
import sys

def get_mistral_response(prompt):
    """Send a request to Mistral via Ollama API."""
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    
    if response and "message" in response and "content" in response["message"]:
        return response["message"]["content"]
    else:
        print("❌ Ollama API Error: Invalid Response")
        return None

def clean_json_response(response):
    """Extract and clean JSON from Mistral's response, fixing common errors."""
    json_match = re.search(r"```json\s*(\[\s*{.*?}\s*])\s*```", response, re.DOTALL)

    if json_match:
        json_block = json_match.group(1).strip()
    else:
        json_start = response.find("[")
        json_end = response.rfind("]") + 1

        if json_start != -1 and json_end != -1:
            json_block = response[json_start:json_end].strip()
        else:
            print("❌ Error: No valid JSON block found in response.")
            raise ValueError("Invalid JSON format")

    # Fix common JSON errors before parsing
    json_block = json_block.replace("'", '"')  # Convert single quotes to double quotes
    json_block = re.sub(r'(\w+):', r'"\1":', json_block)  # Ensure keys are quoted
    json_block = json_block.replace("{,", "{").replace(",}", "}").replace("[,", "[").replace(",]", "]")  # Fix trailing commas
    json_block = json_block.replace("\n", " ")  # Remove unnecessary newlines

    return json_block

# Step 1: Get user input
design_request = input("Describe the UI design you want: ")

# Step 2: Generate multiple design options
prompt = f"""
You are a **highly skilled UI/UX designer and front-end developer**. Based on the user's request: "{design_request}", generate 3 different **realistic, professional website UI concepts**.

Each UI design should include:
1. **Title** - Short, descriptive name for the UI design.
2. **Overall Theme** - Describe the theme (Minimalist, Corporate, Dark Mode, etc.).
3. **Layout Structure** - Explain the page structure (Grid, Sidebar, Hero Section, Cards, etc.).
4. **Color Scheme & Typography** - Provide **primary, secondary, and accent colors**, and suggest **modern fonts**.
5. **Key Components & Features**:
   - Navigation bar (sticky, dropdowns, search)
   - Hero section (image, text, CTA button)
   - Buttons (hover effects, animations)
   - Form inputs (login/signup, search, contact)
   - Cards (profile, features)
   - Modals & Popups (newsletter, alerts)
   - Sidebar (collapsible, animated)
   - Footer (links, copyright, social media)
   - Interactive elements (carousels, tooltips, progress bars)
6. **Animations & Transitions** - Fade-ins, slide-ins, hover effects.
7. **Accessibility Features** - High contrast, keyboard navigation, ARIA roles.
8. **Figma-Style Description** - Exact placement & sizing like a Figma file.

🚨 **STRICT INSTRUCTION** 🚨  
Return the output **ONLY** as a valid JSON array inside a code block (` ```json `). Do NOT include explanations or extra text.
"""

response = get_mistral_response(prompt)

if response:
    try:
        # Debug: Print raw response
        print("\n🛠 Debug - Raw Response:\n", response)

        # Clean and extract valid JSON
        json_block = clean_json_response(response)

        # Try to parse JSON
        design_options = json.loads(json_block)

        # Debug: Print structured response
        print("\n📜 Parsed UI Designs:", json.dumps(design_options, indent=4))

        # Step 3: Display design options
        print("\n🎨 Available UI Designs:\n")
        for idx, design in enumerate(design_options, 1):
            print(f"{idx}. {design.get('Title', 'Untitled Design')}")
            print(f"   📐 Layout: {design.get('Layout Structure', 'Not provided')}")
            print(f"   🎨 Colors: {design.get('Color Scheme', 'Not provided')}")
            print(f"   🔹 Components: {design.get('Components', 'Not provided')}\n")

        # Step 4: Let the user pick a design
        choice = int(input("Enter the number of the design you want: ")) - 1

        if 0 <= choice < len(design_options):
            selected_design = design_options[choice]

            # Step 5: Generate HTML & CSS for the chosen design
            prompt = f"""
            Based on this UI design:
            - **Title**: {selected_design.get('Title', 'Untitled')}
            - **Layout**: {selected_design.get('Layout Structure', 'Unknown')}
            - **Colors**: {selected_design.get('Color Scheme', 'Unknown')}
            - **Components**: {selected_design.get('Components', 'Not provided')}

            🔹 Generate a **fully responsive** HTML & CSS code snippet that follows this design.
            🔹 Ensure the sidebar is styled properly and card widgets are responsive.
            🔹 The color scheme should match the provided details.
            🔹 Use modern HTML5 & CSS3 practices (e.g., Flexbox, Grid).
            🔹 Provide only the complete and functional HTML & CSS code, without explanations.

            🚨 **STRICT INSTRUCTION** 🚨  
            Return the output **ONLY** as a valid HTML file inside a code block (` ```html `).
            """

            ui_code = get_mistral_response(prompt)

            if ui_code:
                # Extract HTML block dynamically
                html_match = re.search(r"```html\s*(.*?)\s*```", ui_code, re.DOTALL)

                if html_match:
                    html_code = html_match.group(1).strip()
                else:
                    print("❌ Error: No valid HTML block found in response. Please try again.")
                    sys.exit(1)

                # Save the generated code
                output_file = "generated_ui.html"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(html_code)

                print(f"\n✅ UI code saved to {output_file}. Open it in a browser to view.")

        else:
            print("❌ Invalid selection. Please restart and choose a valid option.")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Error: Failed to parse the JSON response from Mistral.\n{e}")
        sys.exit(1)
