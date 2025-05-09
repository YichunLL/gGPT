import chainlit as cl
import httpx
import asyncio
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

API_URL = "https://battery-size-cnn.onrender.com/predict/"

# 🔁 Chat memory for DeepSeek follow-ups
chat_history = [
    {
        "role": "system",
        "content": (
            "You are GotionGPT, an expert AI assistant specialized in battery pack and cell optimization. "
            "Explain concepts clearly using plain markdown. Avoid LaTeX formatting (no \\[ \\, \\text{}, \\frac{}). "
            "Do NOT use markdown headings (#). Use bold labels like **Battery Pack Specs**, and format formulas like `E = P × t`."
        )
    }
]

# ✅ Spinner animation loop
async def animate_thinking(msg):
    try:
        i = 0
        while True:
            msg.content = f"🤖 GotionGPT is thinking{'.' * (i % 4)}"
            await msg.update()
            await asyncio.sleep(0.5)
            i += 1
    except asyncio.CancelledError:
        pass

# ✅ Flexible input parser

def parse_input(text: str):
    clean_text = re.sub(r"[，、;|]", ",", text.lower())

    field_patterns = {
        "Length_pack": r"(length|long)\D*(\d+(\.\d+)?)",
        "Width_pack": r"(width|wide)\D*(\d+(\.\d+)?)",
        "Height_pack": r"(height|tall)\D*(\d+(\.\d+)?)",
        "Energy": r"(energy|capacity)\D*(\d+(\.\d+)?)",
        "Total_Voltage": r"(voltage)\D*(\d+(\.\d+)?)"
    }

    extracted = {}
    for key, pattern in field_patterns.items():
        match = re.search(pattern, clean_text)
        if match:
            extracted[key] = float(match.group(2))

    if len(extracted) == 5:
        return extracted

    try:
        numbers = [float(x.strip()) for x in clean_text.split(",") if x.strip()]
        if len(numbers) == 5:
            return {
                "Length_pack": numbers[0],
                "Width_pack": numbers[1],
                "Height_pack": numbers[2],
                "Energy": numbers[3],
                "Total_Voltage": numbers[4]
            }
    except ValueError:
        pass

    return None

@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "🔋 Hi! This is **GotionGPT**, your AI assistant - **NOT limited to** battery cell design and optimization.\n\n"
            "To get started with our self-developed battery size predictor based on the desired pack design\n\n"
            "Enter your input like this:\n"
            "`Length_pack (mm), Width_pack (mm), Height_pack (mm), Energy (kWh), Total Voltage (V)`\n\n"
            "Example: `1000, 1600, 1500, 60, 400`\n\n"
            "Note: I speak both **English** and **Chinese**, so feel free to chat in either!\n"
        )
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        input_data = parse_input(message.content)

        # ✅ Handle Initial Model Prediction
        if input_data:
            analyzing_msg = cl.Message(content="🤖 GotionGPT is analyzing")
            await analyzing_msg.send()
            analyzing_task = asyncio.create_task(animate_thinking(analyzing_msg))

            async with httpx.AsyncClient(timeout=30.0) as client_http:
                try:
                    res = await client_http.post(API_URL, json=input_data)
                    status = res.status_code
                    if status != 200:
                        analyzing_task.cancel()
                        await analyzing_msg.update(
                            content=f"❌ API call failed.\n\n**Status Code:** {status}\n```json\n{res.text}\n```"
                        )
                        return
                    data = res.json()
                except Exception as json_err:
                    analyzing_task.cancel()
                    await analyzing_msg.update(
                        content=f"❌ Could not decode JSON.\n```text\n{res.text}\n```\n**Error:** `{json_err}`"
                    )
                    return

            predictions = data.get("predictions")
            deepseek = data.get("deepseek_analysis", "")

            if not predictions:
                analyzing_task.cancel()
                await analyzing_msg.update(content="❌ The API did not return predictions.")
                return

            try:
                length = float(predictions.get("Length_cell", 0))
                width = float(predictions.get("Width_cell", 0))
                height = float(predictions.get("Height_cell", 0))
                power_density = float(predictions.get("Power_density", 0))
            except (TypeError, ValueError) as e:
                analyzing_task.cancel()
                await analyzing_msg.update(
                    content=f"❌ Prediction data was invalid:\n```python\n{predictions}\n```\n**Error:** {e}"
                )
                return

            analyzing_task.cancel()
            await analyzing_msg.remove()

            pred_msg = (
                f"📐 **Predicted Cell Dimensions from self-developed NN-based predictor**\n"
                f"- Length: {length:.0f} mm\n"
                f"- Width: {width:.0f} mm\n"
                f"- Height: {height:.0f} mm\n"
                f"- Power Density: {power_density:.2f} Wh/kg"
            )
            await cl.Message(content=pred_msg).send()

            if isinstance(deepseek, str):
                chat_history.append({"role": "user", "content": "Analyze this battery pack and predicted cell specs."})
                chat_history.append({"role": "assistant", "content": deepseek})
                await cl.Message(content=deepseek, author="DeepSeek AI").send()
            elif isinstance(deepseek, dict) and "message" in deepseek:
                await cl.Message(content=f"❌ DeepSeek Error:\n\n{deepseek['message']}", author="DeepSeek AI").send()
            else:
                await cl.Message(content="🧠 DeepSeek did not return a valid analysis.").send()
            return

        # 🧠 Follow-up Q&A with DeepSeek
        follow_up = message.content.strip()
        chat_history.append({"role": "user", "content": follow_up})

        thinking_msg = cl.Message(content="🤖 GotionGPT is thinking")
        await thinking_msg.send()
        animation_task = asyncio.create_task(animate_thinking(thinking_msg))

        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="deepseek-chat",
                messages=chat_history,
                max_tokens=500
            )

            reply = response.choices[0].message.content
            chat_history.append({"role": "assistant", "content": reply})

        except Exception as api_err:
            reply = f"❌ DeepSeek follow-up failed:\n```text\n{api_err}```"

        finally:
            animation_task.cancel()
            await thinking_msg.remove()

        await cl.Message(content=reply, author="DeepSeek AI").send()

    except Exception as e:
        import traceback
        traceback.print_exc()
        await cl.Message(
            content=f"⚠️ Unexpected error: `{type(e).__name__}`\nPlease try again or check your server logs."
        ).send()
