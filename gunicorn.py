bind = "0.0.0.0:8000"
workers = 2
threads = 2
timeout = 120
worker_class = "uvicorn.workers.UvicornWorker"
loglevel = "info"
accesslog = "-"
errorlog = "-"
