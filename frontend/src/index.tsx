/* @refresh reload */
import { render } from "solid-js/web";
import "./styles.css";
import App from "./App";
import { holdAlive } from "./api";

// Page-lifetime, not component-lifetime: the server exits with the last window.
holdAlive();

render(() => <App />, document.getElementById("root")!);
