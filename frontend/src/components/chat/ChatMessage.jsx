const ChatMessage = ({ message }) => {
  const isUser = message.role === "user";

  return (
    <div
      className={`message ${
        isUser
          ? "user-message"
          : "assistant-message"
      }`}
    >
      <div className="message-avatar">
        {isUser ? "U" : "AI"}
      </div>

      <div className="message-body">
        <div className="message-label">
          {isUser
            ? "You"
            : "AI Assistant"}
        </div>

        <div className="message-content">
          {message.content}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;