"use client";

import { useScrollAnimation } from "@/hooks/useScrollAnimation";

export default function Process() {
  const { ref, isVisible } = useScrollAnimation();
  const steps = [
    {
      number: "1",
      title: "Image Upload & Preprocessing",
      description: "Medical images are securely uploaded and automatically preprocessed. Our AI normalizes, enhances, and prepares images for optimal analysis.",
      label: "Step 1",
    },
    {
      number: "2",
      title: "Deep Learning Analysis",
      description: "Advanced DenseNet121 model scans every pixel to detect abnormalities, deformities, fractures, lesions, and pathologies across anatomical regions.",
      label: "Step 2",
    },
    {
      number: "3",
      title: "Clinical Report Generation",
      description: "AI generates comprehensive reports with confidence scores, differential diagnosis, and clinical recommendations for radiologist review.",
      label: "Step 3",
    },
  ];

  return (
    <section id="process" className="py-24 px-6 relative">
      <div ref={ref} className={`max-w-7xl mx-auto transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
        <div className="text-center mb-16">
          <p className="text-sm text-accent font-semibold tracking-wider mb-4">PROCESS</p>
          <h2 className="font-display text-5xl md:text-6xl font-bold mb-6">
            How Our AI Analyzes Medical Images
          </h2>
          <p className="text-lg text-muted max-w-2xl mx-auto">
            From image upload to clinical insights in seconds. Our AI-powered workflow 
            delivers accurate, comprehensive analysis to support faster diagnosis and better patient outcomes.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {steps.map((step, index) => (
            <div
              key={index}
              className="bg-card border border-border rounded-2xl p-8 card-hover relative overflow-hidden"
            >
              <div className="absolute top-4 right-4 text-6xl font-bold text-accent/10">
                {step.number}
              </div>
              <div className="relative z-10">
                <p className="text-sm text-accent font-semibold mb-4">{step.label}</p>
                <h3 className="text-2xl font-bold mb-4">{step.title}</h3>
                <p className="text-muted leading-relaxed">{step.description}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-3 gap-8 mt-16 text-center">
          <div>
            <p className="text-4xl font-bold text-accent mb-2">500k+</p>
            <p className="text-sm text-muted">Images Analyzed</p>
          </div>
          <div>
            <p className="text-4xl font-bold text-accent mb-2">95%</p>
            <p className="text-sm text-muted">Detection Accuracy</p>
          </div>
          <div>
            <p className="text-4xl font-bold text-accent mb-2">&lt;2s</p>
            <p className="text-sm text-muted">Analysis Time</p>
          </div>
        </div>
      </div>
    </section>
  );
}
