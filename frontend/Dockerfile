# Development-mode Dockerfile: runs Vite's dev server (matches the
# `command: npm run dev` override in docker-compose.yml). For a production
# build, use a separate multi-stage Dockerfile that runs `npm run build`
# and serves the static output (e.g. with nginx) -- not needed for local
# development, which is this project's primary deployment target.
FROM node:20-alpine

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
