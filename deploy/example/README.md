# Example deployment

A worked example of a Burr site, meant to be copied and edited for a real
observatory.

Burr splits configuration into two pieces:

- **`site.yaml`** (here) — *where* and *what*: coordinates, ASCOM/PWI4 hardware
  addresses, SensorKit entity names, weather thresholds, notifications.
- **a mode preset** (`../presets/*.yaml`) — *how*: which task sources run,
  exposure ranges, and the scheduling balance between them.

They are combined at run time, with the preset deep-merged on top of the site
file.

## Run it

```sh
# SensorKit (NATS-based) control — requires the `sk` extra
burr-sk     --site deploy/example/site.yaml --preset deploy/presets/sda.yaml

# Direct ASCOM hardware control
burr-direct --site deploy/example/site.yaml --preset deploy/presets/survey.yaml

# Safe shutdown
burr-shutdown --site deploy/example/site.yaml --preset deploy/presets/sda.yaml
```

## Make your own

1. Copy this directory: `cp -r deploy/example deploy/<your-site>`.
2. Edit `site.yaml`:
   - set `site.name`, `latitude`, `longitude`, `altitude_km`;
   - set the `hardware` addresses (or `active: false` to simulate a device);
   - for `burr-sk`, fill in the `sk` entity names — discover them with
     `burr-sk --bootstrap --host <agent-host>`;
   - enable `notifications` and set `SLACK_BOT_TOKEN` if you want Slack updates.
3. Pick (or write) a preset in `../presets/` and run with `--site` + `--preset`.

## Simulators

The bundled `deploy/ASCOM.Alpaca.Simulators` and `deploy/PWI4` Docker setups let
you exercise the full pipeline without real hardware — point the `hardware`
addresses in `site.yaml` at them.
