import { t } from "../i18n";

/** The (?) that reveals the explanatory prose (one global preference, kept in
    localStorage by App). `warn` tints it when what is hidden behind it includes
    a warning. `type=button`: inside the Settings dialog every other button
    submits the <form method="dialog"> and closes it. */
export function HelpToggle(props: { open: boolean; warn?: boolean; onToggle: () => void }) {
  const label = () => (props.open ? t().help.hide : props.warn ? t().help.showWarn : t().help.show);
  return (
    <button
      type="button"
      classList={{ help: true, warn: !props.open && !!props.warn, on: props.open }}
      title={label()}
      aria-label={label()}
      aria-expanded={props.open}
      onClick={props.onToggle}
    >
      ?
    </button>
  );
}
