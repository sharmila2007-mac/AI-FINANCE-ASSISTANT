const toggle = document.getElementById("chat-toggle");
const panel = document.getElementById("chat-panel");
if (toggle && panel) {
  toggle.addEventListener("click", () => panel.classList.toggle("hidden"));
  document.getElementById("chat-close").addEventListener("click", () => panel.classList.add("hidden"));
  document.getElementById("chat-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = document.getElementById("chat-input");
    const messages = document.getElementById("chat-messages");
    const message = input.value.trim();
    if (!message) return;
    const userLine = document.createElement("p");
    userLine.className = "user-msg";
    userLine.textContent = message;
    messages.appendChild(userLine);
    input.value = "";
    try {
      const response = await fetch("/chatbot", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message})
      });
      const data = await response.json();
      const botLine = document.createElement("p");
      botLine.className = "bot-msg";
      botLine.textContent = data.answer || data.error || "Sorry, I couldn't answer that.";
      messages.appendChild(botLine);
      messages.scrollTop = messages.scrollHeight;
    } catch {
      const errorLine = document.createElement("p");
      errorLine.className = "bot-msg";
      errorLine.textContent = "Unable to connect. Please try again.";
      messages.appendChild(errorLine);
    }
  });
}
