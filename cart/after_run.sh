#!/usr/bin/env bash
# Stand-in for the tail of run_cart.sh when that host script is gone (2026-10-21 13:24: systemd-oomd killed the whole
# user session, run_cart.sh with it, while the run's container kept going).  Runs in a container as root so a later
# session kill cannot stop it: waits for the run log's "done -> " and "manifest -> " lines, chowns the scene to UID:GID,
# writes Data/Cart/done/run_f120.
#   docker run -d -v /:/host -e UID_GID=1000:1000 scsam3 bash /host<repo>/cart/after_run.sh /host<repo>
set -uo pipefail
ROOT="$1"; C="${ROOT}/Data/Cart"; KEY=cart_f120; LOG="${C}/logs/run_f120.log"
until grep -aq "^done -> " "${LOG}" && grep -aq "manifest -> " "${LOG}"; do sleep 60; done
sleep 30
chown -R "${UID_GID}" "${C}/${KEY}"
touch "${C}/done/run_f120" && chown "${UID_GID}" "${C}/done/run_f120"
echo "$(date '+%F %T') run finished; chowned, ${C}/done/run_f120 written" >> "${C}/logs/chain.log"
chown "${UID_GID}" "${C}/logs/chain.log"
