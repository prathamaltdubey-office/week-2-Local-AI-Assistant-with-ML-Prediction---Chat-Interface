const Header = () => {
  return (
    <header className="header glass">
      <div className="brand">
        <div className="brand-icon">
          ◈
        </div>

        <div>
          <h1>Customer Churn AI</h1>

          <p>
            ML-powered customer intelligence platform
          </p>
        </div>
      </div>

      <div className="status-pill">
        <span className="status-dot"></span>

        <span>AI Assistant</span>

        <span className="status-live">
          LIVE
        </span>
      </div>
    </header>
  );
};

export default Header;