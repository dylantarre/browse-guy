FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    netcat-traditional \
    gnupg \
    curl \
    unzip \
    xvfb \
    libgconf-2-4 \
    libxss1 \
    libnss3 \
    libnspr4 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    xdg-utils \
    fonts-liberation \
    dbus \
    dbus-x11 \
    xauth \
    xvfb \
    x11vnc \
    tigervnc-tools \
    supervisor \
    net-tools \
    procps \
    git \
    python3-numpy \
    fontconfig \
    fonts-dejavu \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    && rm -rf /var/lib/apt/lists/*

# Install D-Bus and configure it
RUN apt-get update && apt-get install -y \
    dbus \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/dbus \
    && dbus-uuidgen > /var/lib/dbus/machine-id

# Add D-Bus configuration
RUN mkdir -p /etc/dbus-1/system.d && \
    echo '<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-BUS Bus Configuration 1.0//EN" \
    "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd"> \
    <busconfig> \
    <type>system</type> \
    <allow_anonymous/> \
    <policy context="default"> \
    <allow send_destination="*" eavesdrop="true"/> \
    <allow eavesdrop="true"/> \
    <allow own="*"/> \
    </policy> \
    </busconfig>' > /etc/dbus-1/system.conf

# Install noVNC
RUN git clone https://github.com/novnc/noVNC.git /opt/novnc \
    && git clone https://github.com/novnc/websockify /opt/novnc/utils/websockify \
    && ln -s /opt/novnc/vnc.html /opt/novnc/index.html

# Set platform for ARM64 compatibility
ARG TARGETPLATFORM=linux/amd64

# Set up working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and browsers with system dependencies
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install --with-deps chromium
RUN playwright install-deps

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BROWSER_USE_LOGGING_LEVEL=info
ENV CHROME_PATH=/ms-playwright/chromium-*/chrome-linux/chrome
ENV ANONYMIZED_TELEMETRY=false
ENV DISPLAY=:99
ENV RESOLUTION=1920x1080x24
ENV VNC_PASSWORD=vncpassword
ENV CHROME_PERSISTENT_SESSION=true
ENV RESOLUTION_WIDTH=1920
ENV RESOLUTION_HEIGHT=1080

# Set up Chrome profile directory
RUN rm -rf /root/.config/chromium /root/.cache/chromium /tmp/chrome* && \
    mkdir -p /root/.config/chromium && \
    chmod -R 777 /root/.config/chromium && \
    mkdir -p /root/.cache/chromium && \
    chmod -R 777 /root/.cache/chromium && \
    mkdir -p /app/data/chrome_data && \
    chmod -R 777 /app/data/chrome_data

# Set up supervisor configuration
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Add dbus service to supervisord config
RUN echo "[program:dbus]\ncommand=/usr/bin/dbus-daemon --system --nofork\nautorestart=true\npriority=1\nstartretries=3\nstartsecs=3" >> /etc/supervisor/conf.d/supervisord.conf

# Add cleanup script
RUN echo '#!/bin/bash\npkill -9 -f chrome || true\nrm -rf /tmp/chrome*\nrm -rf ~/.config/chromium\nrm -rf ~/.cache/chromium\nrm -f /tmp/.X*-lock\nrm -f /tmp/.X11-unix/X*\ndbus-uuidgen > /var/lib/dbus/machine-id\nmkdir -p /var/run/dbus\ndbus-daemon --system --fork' > /usr/local/bin/cleanup-chrome && \
    chmod +x /usr/local/bin/cleanup-chrome

EXPOSE 7789 6081 5902 9223

# Run cleanup and start supervisord
ENTRYPOINT ["/bin/bash", "-c", "/usr/local/bin/cleanup-chrome && supervisord -n -c /etc/supervisor/supervisord.conf"]
