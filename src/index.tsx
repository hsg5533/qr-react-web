import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import reportWebVitals from "./reportWebVitals";
import Points from "./Points";

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);
root.render(<Points />);

reportWebVitals();
