FROM node:18-alpine

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

# Set API URL for external access
ENV VITE_API_URL=http://localhost:9001

EXPOSE 3000

# Use dev server for live reloading
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]