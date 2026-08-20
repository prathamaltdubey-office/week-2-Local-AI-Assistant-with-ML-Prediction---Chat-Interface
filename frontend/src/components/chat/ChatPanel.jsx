import useChat from "../../hooks/useChat";

import MessageList from "./MessageList";
import ChatInput from "./ChatInput";

const ChatPanel = () => {
  const {
    messages,
    input,
    setInput,
    loading,
    sendMessage,
  } = useChat();

  return (
    <section className="chat-section glass">
      <div className="panel-header">
        <div className="panel-title-row">
          <div className="panel-icon chat-icon">
            ✦
          </div>

          <div>
            <h2>
              AI Chat Assistant
            </h2>

            <p>
              Ask questions about customer churn,
              the ML project, or predictions.
            </p>
          </div>
        </div>

        <div className="mini-status">
          <span></span>
          Online
        </div>
      </div>

      <div className="chat-container">
        <MessageList
          messages={messages}
          loading={loading}
        />

        <ChatInput
          input={input}
          setInput={setInput}
          loading={loading}
          sendMessage={sendMessage}
        />
      </div>
    </section>
  );
};

export default ChatPanel;