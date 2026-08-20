import { useState } from "react";
import { sendChatMessage } from "../services/api";

const initialMessage = {
  role: "assistant",
  content:
    "Hello! I'm your Customer Churn AI Assistant. Ask me about customer churn, the ML project, or provide customer information for a churn prediction.",
};

const useChat = () => {
  const [messages, setMessages] = useState([
    initialMessage,
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    const message = input.trim();

    if (!message || loading) {
      return;
    }

    setMessages((previousMessages) => [
      ...previousMessages,
      {
        role: "user",
        content: message,
      },
    ]);

    setInput("");
    setLoading(true);

    try {
      const response = await sendChatMessage(message);

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          content:
            response.answer ||
            "I couldn't generate a response.",
        },
      ]);
    } catch (error) {
      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "assistant",
          content:
            error.message ||
            "Sorry, something went wrong while contacting the AI service.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return {
    messages,
    input,
    setInput,
    loading,
    sendMessage,
  };
};

export default useChat;