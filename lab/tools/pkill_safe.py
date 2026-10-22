"""Kill processes whose cmdline contains a pattern, excluding this process and its ancestors."""
import os, sys, signal
pat = sys.argv[1]
me = os.getpid(); anc = set()
p = me
while p > 1:
    anc.add(p)
    try:
        p = int(open(f"/proc/{p}/stat").read().split(")")[1].split()[1])
    except Exception:
        break
for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) in anc:
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    except Exception:
        continue
    if pat in cmd and "pkill_safe" not in cmd:
        os.kill(int(pid), signal.SIGTERM); print("killed", pid, cmd[:80])
