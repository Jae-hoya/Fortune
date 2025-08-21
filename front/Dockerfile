# syntax=docker/dockerfile:1

# 1) Install dependencies
FROM node:20-bookworm-slim AS deps
WORKDIR /app

# Copy minimal files for dependency install
COPY package.json package-lock.json* ./

# Use npm (no lockfile assumed); if lockfile exists it will be used
RUN npm install


# 2) Build the Next.js app
FROM node:20-bookworm-slim AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

# Pass client env to build (NEXT_PUBLIC_* must be present at build time)
ARG NEXT_PUBLIC_WEBSOCKET_URL
ENV NEXT_PUBLIC_WEBSOCKET_URL=${NEXT_PUBLIC_WEBSOCKET_URL}

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build production assets
RUN npm run build

# Prune devDependencies for a smaller runtime image
RUN npm prune --omit=dev || true


# 3) Create a minimal runtime image
FROM node:20-bookworm-slim AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

# Create non-root user for security
RUN useradd -m nextjs && chown -R nextjs:nextjs /app

# Copy only the necessary runtime files
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/next.config.ts ./next.config.ts
COPY --from=builder /app/node_modules ./node_modules

USER nextjs

EXPOSE 3000

# Bind to provided PORT and 0.0.0.0 for platforms like Vercel
CMD ["sh", "-c", "npm run start -- -p ${PORT} -H 0.0.0.0"]


