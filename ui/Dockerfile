FROM node:18-alpine

# Accept port as build argument and convert to environment variable
ARG UI_PORT=7905
ENV UI_PORT=${UI_PORT}

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

# Set API URL for external access (will be overridden by docker-compose env)
ENV VITE_API_URL=http://localhost:7901

# Expose the configurable port
EXPOSE ${UI_PORT}

# Use dev server for live reloading with configurable port
CMD ["sh", "-c", "npm run dev -- --host 0.0.0.0 --port ${UI_PORT}"]