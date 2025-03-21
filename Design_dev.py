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
        print("\n❌ Ollama API Error: Invalid Response")
        return None

def extract_json(response):
    """Extract and clean JSON from the response."""
    json_match = re.search(r"```json\s*\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        json_text = json_match.group(1).strip()
    else:
        json_start = response.find("[")
        json_end = response.rfind("]") + 1
        if json_start != -1 and json_end != -1:
            json_text = response[json_start:json_end].strip()
        else:
            print("❌ Error: No valid JSON block found in response.")
            raise ValueError("Invalid JSON format")

    # 🚨 Remove invalid placeholders like `...`
    json_text = re.sub(r'"\.\.\."', '"Not provided"', json_text)  

    # Fix common JSON issues (trailing commas)
    json_text = re.sub(r",\s*([}\]])", r"\1", json_text)  

    try:
        return json.loads(json_text)  # Validate JSON
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}\nRaw JSON: {json_text}")
        raise ValueError("Failed to parse JSON")


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

        # Extract and fix JSON
        design_options = extract_json(response)

        # Step 3: Display design options
        print("\n🎨 Available UI Designs:\n")
        for idx, design in enumerate(design_options, 1):
            print(f"{idx}. {design.get('Title', 'Untitled Design')}")
            print(f"   📐 Layout: {design.get('Layout Structure', 'Not provided')}")
            print(f"   🎨 Colors: {json.dumps(design.get('Color Scheme & Typography', {}), indent=2)}")
            print(f"   🔹 Components: {json.dumps(design.get('Key Components & Features', {}), indent=2)}")

        # Step 4: Let the user pick a design
        choice = int(input("Enter the number of the design you want: ")) - 1

        if 0 <= choice < len(design_options):
            selected_design = design_options[choice]
            print(f"Selected Design: {selected_design}")


            # Step 5: Generate full HTML & inline CSS for the chosen design
            prompt = f"""
            Based on this UI design:
            - **Title**: {selected_design.get('Title', 'Untitled')}
            - **Layout**: {selected_design.get('Layout Structure', 'Unknown')}
            - **Colors**: {selected_design.get('Color Scheme & Typography', 'Unknown')}
            - **Components**: {selected_design.get('Key Components & Features', 'Not provided')}
            
            🔹 Generate a **fully functional HTML file** with **INLINE CSS** (inside the `<style>` tag).  
            🔹 The UI should be **fully structured and visually styled**.  
            🔹 **DO NOT** include any external images, URLs, or links.  

            🚨 **STRICT FORMAT REQUIREMENTS** 🚨  
            - **Return a fully valid HTML file** with:
                - **Inline CSS (inside `<style>`)**
                - **All required sections (Navbar, Hero, Cards, Footer)**
                - **NO broken or incomplete HTML**
            - **Wrap the HTML inside a code block (` ```html `) and do NOT include explanations.**
            - If any section is missing, **regenerate the response**.

            Example Expected Output:
            ```html
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <style>
                body {{ background-color: #2d3748; color: #f9cf46; font-family: 'Open Sans', sans-serif; margin: 0; }}
                .navbar {{ background-color: #1a202c; padding: 15px; text-align: center; }}
                .hero {{ text-align: center; padding: 100px 20px; }}
                .btn {{ background-color: #e74c3c; color: white; padding: 10px 20px; border: none; }}
                .btn:hover {{ background-color: #ff5733; }}
              </style>
            </head>
            <body>
              <div class="navbar"><a href="#">Home</a> <a href="#">Shop</a> <a href="#">Categories</a> <a href="#">Contact</a></div>
              <div class="hero"><h1>Welcome to Dark Professional</h1><p>Enhance your skills with our courses.</p><button class="btn">Get Started</button></div>
            </body>
            </html>
            ```
            """

            ui_code = get_mistral_response(prompt)

            if ui_code and "```html" in ui_code:
                html_match = re.search(r"```html\s*\n(.*?)\n```", ui_code, re.DOTALL)

                if html_match:
                    html_code = html_match.group(1).strip()

                    with open("generated_ui.html", "w", encoding="utf-8") as f:
                        f.write(html_code)

                    print("\n✅ UI code saved to generated_ui.html. Open it in a browser to view.")

                else:
                    print("❌ Error: No valid HTML block found. Full response:", ui_code)
                    sys.exit(1)

    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
