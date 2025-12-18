"use client";

import { useState, useRef } from "react";
import { AuthService } from "@/lib/auth";
import { useRouter } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

interface Prediction {
  class: string;
  confidence: number;
}

interface PredictionResult {
  prediction: string;
  confidence: number;
  top_predictions: Prediction[];
  prediction_id: number;
  heatmap?: string; // Base64 encoded heatmap
}

export default function MedicalAnalysis() {
  const router = useRouter();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [modality, setModality] = useState<'xray' | 'ct' | 'ultrasound'>('xray'); // Default X-Ray
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const modalities = [
    { id: 'xray', label: 'Chest X-Ray', icon: '🩻' },
    { id: 'ct', label: 'CT Scan', icon: '🧠' },
    { id: 'ultrasound', label: 'Ultrasound', icon: '📡' },
  ];

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
        setResult(null);
        setError("");
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage || !fileInputRef.current?.files?.[0]) {
      setError("Please select an image first");
      return;
    }

    const token = AuthService.getToken();
    if (!token) {
      setError("Please login to analyze images");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append('image', fileInputRef.current.files[0]);
      formData.append('modality', modality); // Send selected modality

      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        if (response.status === 401) {
          AuthService.logout();
          router.push('/');
          throw new Error("Session expired. Please login again.");
        }
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze image');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setResult(null);
    setError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <section className="py-24 px-6 min-h-screen">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="font-display text-5xl md:text-6xl font-bold mb-6">
            AI-Powered <span className="text-accent">Diagnostic Imaging</span>
          </h1>
          <p className="text-lg text-muted max-w-2xl mx-auto">
            Upload medical images for comprehensive AI analysis. Our deep learning models 
            automatically detect abnormalities, deformities, and pathologies with clinical-grade accuracy.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-card border border-border rounded-2xl p-8">
            
            {/* Modality Selector */}
            <h3 className="text-xl font-bold mb-4">Select Scan Type</h3>
            <div className="grid grid-cols-3 gap-2 mb-8">
              {modalities.map((m) => (
                <button
                  key={m.id}
                  onClick={() => {
                    setModality(m.id as any);
                    setResult(null); // Clear previous results on switch
                  }}
                  className={`p-3 rounded-xl border transition-all flex flex-col items-center gap-2
                    ${modality === m.id 
                      ? 'bg-accent text-white border-accent shadow-lg scale-105' 
                      : 'bg-background border-border hover:border-accent/50 text-muted hover:text-foreground'
                    }`}
                >
                  <span className="text-2xl">{m.icon}</span>
                  <span className="text-xs font-bold">{m.label}</span>
                </button>
              ))}
            </div>

            <h3 className="text-2xl font-bold mb-6">Upload Image</h3>
            
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-border rounded-xl p-8 text-center cursor-pointer hover:border-accent transition-colors mb-6"
            >
              {selectedImage ? (
                <img
                  src={selectedImage}
                  alt="Selected medical image"
                  className="max-h-64 mx-auto rounded-lg"
                />
              ) : (
                <div className="py-12">
                  <svg
                    className="w-16 h-16 mx-auto mb-4 text-muted"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  <p className="text-muted">Click to upload medical image</p>
                  <p className="text-sm text-muted mt-2">PNG, JPG up to 10MB</p>
                </div>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              className="hidden"
            />

            <div className="flex gap-4">
              <button
                onClick={handleAnalyze}
                disabled={!selectedImage || loading}
                className="flex-1 bg-accent text-background py-3 rounded-full font-semibold hover:bg-accent/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Analyzing..." : "Analyze Image"}
              </button>
              
              {selectedImage && (
                <button
                  onClick={handleClear}
                  className="px-6 py-3 border border-border rounded-full font-semibold hover:border-accent transition-colors"
                >
                  Clear
                </button>
              )}
            </div>

            {error && (
              <div className="mt-4 bg-red-500/10 border border-red-500/50 rounded-lg p-3 text-red-500 text-sm">
                {error}
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-card border border-border rounded-2xl p-8">
            <h3 className="text-2xl font-bold mb-6">Analysis Results</h3>
            
            {result ? (
              <div className="space-y-6">
                <div className="bg-accent/10 border border-accent rounded-xl p-6">
                  <p className="text-sm text-accent font-semibold mb-2">🔍 Primary Finding</p>
                  <p className="text-3xl font-bold mb-2">{result.prediction}</p>
                  <div className="flex items-center gap-2 mb-3">
                    <div className="flex-1 bg-background rounded-full h-3">
                      <div
                        className="bg-accent h-3 rounded-full transition-all"
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-lg font-semibold text-accent">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-sm text-muted">
                    {result.confidence > 0.9 ? "High confidence detection" : 
                     result.confidence > 0.7 ? "Moderate confidence - review recommended" :
                     "Low confidence - manual review required"}
                  </p>
                </div>

                {/* Grad-CAM Visualization */}
                {result.heatmap && (
                  <div className="bg-background rounded-xl p-6 border border-border">
                    <div className="flex items-center gap-2 mb-4">
                      <span className="text-xl">👁️</span>
                      <h4 className="font-bold text-foreground">AI Vision (Explainability)</h4>
                    </div>
                    <div className="relative rounded-lg overflow-hidden border border-border group">
                      <img 
                        src={`data:image/jpeg;base64,${result.heatmap}`} 
                        alt="AI Attention Heatmap"
                        className="w-full h-auto transform group-hover:scale-105 transition-transform duration-500"
                      />
                      <div className="absolute bottom-0 left-0 right-0 bg-black/60 backdrop-blur-sm p-3">
                        <p className="text-xs text-white text-center">
                          Heatmap shows regions of interest (Red = High Attention)
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div>
                  <p className="text-sm font-semibold mb-4">📊 Differential Diagnosis</p>
                  <div className="space-y-3">
                    {result.top_predictions.map((pred, index) => (
                      <div key={index} className="bg-background rounded-lg p-4 border border-border hover:border-accent/50 transition-colors">
                        <div className="flex justify-between items-center mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-muted">#{index + 1}</span>
                            <span className="font-medium">{pred.class}</span>
                          </div>
                          <span className="text-accent font-semibold">
                            {(pred.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-border rounded-full h-2">
                          <div
                            className="bg-accent h-2 rounded-full transition-all"
                            style={{ width: `${pred.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="bg-background rounded-lg p-4 border border-border">
                    <p className="font-semibold text-sm mb-2">📋 Clinical Recommendations</p>
                    <ul className="text-sm text-muted space-y-1">
                      <li>• Correlate with clinical history and symptoms</li>
                      <li>• Consider additional imaging if needed</li>
                      <li>• Follow institutional protocols for findings</li>
                    </ul>
                  </div>
                  
                  <div className="bg-accent/5 border border-accent/20 rounded-xl p-4 flex gap-4 items-start">
                    <div className="bg-accent/10 p-2 rounded-full shrink-0">
                      <svg className="w-5 h-5 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <h5 className="font-semibold text-foreground text-sm mb-1">Clinical Support Tool</h5>
                      <p className="text-xs text-muted leading-relaxed">
                        This AI analysis is intended to assist medical professionals in triage and prioritization. 
                        It does not constitute a final diagnosis. Please verify all findings with a certified radiologist.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-center text-muted">
                <div>
                  <svg
                    className="w-16 h-16 mx-auto mb-4 opacity-50"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <p>Upload and analyze an image to see results</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
