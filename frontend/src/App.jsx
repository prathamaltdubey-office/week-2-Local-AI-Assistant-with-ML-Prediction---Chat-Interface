import "./App.css";

import Header from "./components/common/Header";
import Footer from "./components/common/Footer";

import ChatPanel from "./components/chat/ChatPanel";
import PredictionPanel from "./components/prediction/PredictionPanel";

function App() {
  return (
    <div className="app">
      <div className="background-orb orb-one"></div>
      <div className="background-orb orb-two"></div>
      <div className="background-orb orb-three"></div>

      <Header />

      <main className="main-content">
        <ChatPanel />

        <PredictionPanel />
      </main>

      <Footer />
    </div>
  );
}

export default App;
