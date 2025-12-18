# Neuron AI - Copilot Instructions

## Project Context
Neuron AI is a professional-grade medical imaging decision support system. It uses a **Dual-Stream Ensemble Architecture** (EfficientNet-B4 + DenseNet169) to analyze Chest X-Rays for 9 specific pathologies.

## Architecture & Stack

### Backend (Root)
- **Framework:** Python / Flask (`api_server.py`)
- **ML Engine:** PyTorch (`torch`, `torchvision`, `timm`)
- **Database:** SQLite (`medical_ai.db`)
- **Key Files:**
  - `api_server.py`: Main API entry point, handles inference and auth.
  - `weights/`: Contains model weights (`xray_efficientnet.pth`, `xray_densenet.pth`).

### Frontend (`landeros-clone/`)
- **Framework:** Next.js 14 (App Router)
- **Styling:** Tailwind CSS
- **Language:** TypeScript
- **Key Directories:**
  - `app/`: App Router pages and layouts.
  - `components/`: Reusable UI components.
  - `lib/`: Utilities (e.g., `auth.ts`).

## Key Workflows

### Running the Application
- **Backend:**
  ```bash
  # Activate environment (if applicable)
  source medical_env/bin/activate
  # Run server
  python api_server.py
  ```
- **Frontend:**
  ```bash
  cd landeros-clone
  npm run dev
  ```

### AI/ML Development
- **Inference Logic:**
  - Images are resized to **224x224**.
  - Predictions are an average of EfficientNet and DenseNet outputs (Softmax Averaging).
  - **Confidence Threshold:** 0.55 (Predictions below this are flagged).
- **Classes:** 9 classes including Pneumonia, Normal, Cardiomegaly, etc.

## Code Style & Conventions

### Python (Backend)
- Use `snake_case` for functions and variables.
- Type hinting is encouraged but not strictly enforced in legacy scripts.
- **Error Handling:** Wrap API endpoints in try-except blocks and return JSON errors.
- **Security:** Never commit `SECRET_KEY` or real patient data. Use environment variables.

### TypeScript/React (Frontend)
- Use `PascalCase` for components.
- Prefer **Functional Components** with Hooks.
- Use **Tailwind CSS** for styling (avoid CSS modules unless necessary).
- **Next.js:** Use `next/image` for image optimization. Use Server Components by default, add `'use client'` only when interactivity is needed.

## Critical Integration Points
- **API Communication:** Frontend calls Backend at `http://localhost:5000` (default Flask port).
- **CORS:** Enabled on Backend to allow requests from Frontend.
- **Auth:** JWT based authentication.

## Specific Patterns
- **Ensemble Inference:** When modifying inference logic, ensure both models are loaded and their outputs are averaged.
- **Grad-CAM:** The system supports explainability via Grad-CAM. Ensure any model changes maintain compatibility with the Grad-CAM hooks.
