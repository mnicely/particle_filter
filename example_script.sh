#!/bin/bash
declare -a APP='./filters'
declare -a MCS=5
declare -a SAMPLES=250

# Run with default settings
${APP}
${APP} -g

# Run with serial version in the background
${APP} &
pid0=$!
${APP} -g
wait ${pid0}

# Run with using truth data and other options
${APP} -t -r0 -m${MCS} -s${SAMPLES} &
pid0=$!

${APP} -t -r1 -m${MCS} -s${SAMPLES} &
pid1=$!

${APP} -g -t -r0 -m${MCS} -s${SAMPLES}
${APP} -g -t -r1 -m${MCS} -s${SAMPLES}
${APP} -g -t -r2 -m${MCS} -s${SAMPLES}

${APP} -g -t -r0 -p1048576 -m${MCS} -s${SAMPLES}
${APP} -g -t -r1 -p1048576 -m${MCS} -s${SAMPLES}
${APP} -g -t -r2 -p1048576 -m${MCS} -s${SAMPLES}

wait ${pid0}
wait ${pid1}
