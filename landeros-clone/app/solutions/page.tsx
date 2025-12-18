"use client";

import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function SolutionsPage() {
  return (
    <main className="min-h-screen relative">
      <div className="relative z-10">
        <Navbar />
        
        <div className="pt-32 pb-20 px-6">
          <div className="max-w-7xl mx-auto">
            {/* Header */}
            <div className="text-center mb-16">
              <h1 className="text-6xl font-bold mb-6">
                Complete <span className="text-accent">AI Solutions</span>
              </h1>
              <p className="text-xl text-muted max-w-3xl mx-auto">
                End-to-end medical imaging AI solutions for hospitals, clinics, and healthcare providers
              </p>
            </div>

            {/* Main Solutions */}
            <div className="space-y-12 mb-20">
              <div className="bg-white rounded-3xl p-12 border border-border card-hover">
                <div className="grid md:grid-cols-2 gap-12 items-center">
                  <div>
                    <span className="inline-block bg-accent/10 text-accent px-4 py-2 rounded-full text-sm font-semibold mb-4">
                      Core Solution
                    </span>
                    <h2 className="text-4xl font-bold mb-4">Automated Abnormality Detection</h2>
                    <p className="text-muted text-lg mb-6">
                      Our AI algorithms automatically scan medical images pixel-by-pixel to identify abnormalities, 
                      deformities, fractures, lesions, masses, and nodules across multiple anatomical regions.
                    </p>
                    <ul className="space-y-3">
                      <li className="flex items-start gap-3">
                        <svg className="w-6 h-6 text-accent flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>Real-time detection in under 2 seconds</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <svg className="w-6 h-6 text-accent flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>95%+ accuracy across 8 pathology types</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <svg className="w-6 h-6 text-accent flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>Confidence scoring for clinical decision support</span>
                      </li>
                    </ul>
                  </div>
                  <div className="bg-gradient-to-br from-accent/20 to-accent-dark/20 rounded-2xl p-12 text-center">
                    <div className="text-8xl mb-4">🔍</div>
                    <p className="text-2xl font-bold">Pixel-Perfect Analysis</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-3xl p-12 border border-border card-hover">
                <div className="grid md:grid-cols-2 gap-12 items-center">
                  <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl p-12 text-center order-2 md:order-1">
                    <div className="text-8xl mb-4">🧠</div>
                    <p className="text-2xl font-bold">Deep Learning Power</p>
                  </div>
                  <div className="order-1 md:order-2">
                    <span className="inline-block bg-purple-500/10 text-purple-600 px-4 py-2 rounded-full text-sm font-semibold mb-4">
                      AI Technology
                    </span>
                    <h2 className="text-4xl font-bold mb-4">DenseNet121 Architecture</h2>
                    <p className="text-muted text-lg mb-6">
                      Powered by state-of-the-art DenseNet121 deep learning model trained on millions of medical 
                      images for precise pathology detection and classification.
                    </p>
                    <ul className="space-y-3">
                      <li className="flex items-start gap-3">
                        <svg className="w-6 h-6 text-purple-600 flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>121 convolutional layers for deep feature extraction</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <svg className="w-6 h-6 text-purple-600 flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>Trained on 100,000+ annotated medical images</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <svg className="w-6 h-6 text-purple-600 flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>Continuous learning and model updates</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-3xl p-12 border border-border card-hover">
                <div className="grid md:grid-cols-2 gap-12 items-center">
                  <div>
                    <span className="inline-block bg-green-500/10 text-green-600 px-4 py-2 rounded-full text-sm font-semibold mb-4">
                      Clinical Integration
                    </span>
                    <h2 className="text-4xl font-bold mb-4">Clinical Decision Support</h2>
                    <p className="text-muted text-lg mb-6">
                      AI-generated insights help radiologists prioritize critical cases, reduce diagnostic errors, 
                      and provide comprehensive differential diagnosis with confidence scoring.
                    </p>
                    <ul className="space-y-3">
                      <li className="flex items-start gap-3">
                        <svg className="w-6 h-6 text-green-600 flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>Top-3 differential diagnosis ranking</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <svg className="w-6 h-6 text-green-600 flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>Clinical recommendations and next steps</span>
                      </li>
                      <li className="flex items-start gap-3">
                        <svg className="w-6 h-6 text-green-600 flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>Automated report generation</span>
                      </li>
                    </ul>
                  </div>
                  <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-2xl p-12 text-center">
                    <div className="text-8xl mb-4">⚕️</div>
                    <p className="text-2xl font-bold">Clinical Excellence</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Pathologies Grid */}
            <div className="bg-white rounded-3xl p-12 border border-border mb-20">
              <h2 className="text-4xl font-bold text-center mb-12">Supported Pathologies</h2>
              <div className="grid md:grid-cols-4 gap-6">
                {[
                  { name: "Atelectasis", icon: "🫁", desc: "Lung collapse detection" },
                  { name: "Cardiomegaly", icon: "❤️", desc: "Enlarged heart identification" },
                  { name: "Effusion", icon: "💧", desc: "Fluid buildup analysis" },
                  { name: "Infiltration", icon: "🔬", desc: "Tissue density abnormalities" },
                  { name: "Mass", icon: "⚫", desc: "Abnormal growth detection" },
                  { name: "Nodule", icon: "🔴", desc: "Small opacity analysis" },
                  { name: "Pneumonia", icon: "🦠", desc: "Infection identification" },
                  { name: "Pneumothorax", icon: "💨", desc: "Air in chest cavity" },
                ].map((pathology, i) => (
                  <div key={i} className="bg-background-dark p-6 rounded-2xl text-center card-hover">
                    <div className="text-5xl mb-3">{pathology.icon}</div>
                    <h3 className="font-bold mb-2">{pathology.name}</h3>
                    <p className="text-sm text-muted">{pathology.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* CTA */}
            <div className="bg-gradient-to-br from-accent to-accent-dark text-white rounded-3xl p-12 text-center">
              <h2 className="text-4xl font-bold mb-4">Ready to Transform Your Practice?</h2>
              <p className="text-white/80 text-lg mb-8 max-w-2xl mx-auto">
                Join leading healthcare providers using AI to improve diagnosis accuracy and patient outcomes
              </p>
              <div className="flex gap-4 justify-center">
                <a href="/analyze" className="bg-white text-accent px-8 py-4 rounded-full font-semibold hover:bg-white/90 transition-all">
                  Try It Now
                </a>
                <a href="/documentation" className="bg-white/10 border-2 border-white text-white px-8 py-4 rounded-full font-semibold hover:bg-white/20 transition-all">
                  View Documentation
                </a>
              </div>
            </div>
          </div>
        </div>

        <Footer />
      </div>
    </main>
  );
}
