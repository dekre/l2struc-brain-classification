if [ -f jupyter.pid ]; then
    source jupyter.pid
    echo "Terminating Jupyter Notebook with PID=${JPID}"
    kill -9 ${JPID}
else
    echo "No process id found. To check if the notebook is running please execute: jupyter notebook list"
fi


