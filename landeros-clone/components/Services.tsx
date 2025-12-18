"use client";

export default function Services() {
  const benefits = [
    {
      title: "Automated Detection",
      description: "AI algorithms automatically scan and identify abnormalities, deformities, and critical findings in medical images with clinical-grade accuracy.",
    },
    {
      title: "Instant Results",
      description: "Get comprehensive analysis in seconds. Reduce diagnosis time from hours to minutes while maintaining high accuracy standards.",
    },
  ];

  const services = [
    {
      title: "Comprehensive Abnormality Detection",
      description: "Our AI scans every pixel to detect fractures, lesions, masses, nodules, and other deformities across multiple anatomical regions.",
      tag: "Real-time Analysis",
    },
    {
      title: "Deep Learning Model Architecture",
      description: "Powered by state-of-the-art DenseNet121 trained on millions of medical images for precise pathology detection.",
      code: true,
    },
    {
      title: "Clinical Decision Support",
      description: "AI-generated insights help radiologists prioritize critical cases and reduce diagnostic errors with confidence scoring.",
      tag: "FDA-Ready Algorithms",
    },
  ];

  const features = [
    "Fracture Detection",
    "Pneumonia Identification",
    "Tumor & Mass Detection",
    "Nodule Analysis",
    "Cardiomegaly Detection",
    "Pleural Effusion",
    "Atelectasis Recognition",
    "Multi-Pathology Screening",
  ];

  return (
    <section id="services" className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="font-display text-5xl md:text-6xl font-bold mb-12">
            Medical Imaging AI Solutions
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto mb-16">
            {benefits.map((benefit, index) => (
              <div key={index} className="text-left">
                <h3 className="text-2xl font-bold mb-3">{benefit.title}</h3>
                <p className="text-muted">{benefit.description}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-8">
          {services.map((service, index) => (
            <div
              key={index}
              className="bg-card border border-border rounded-2xl p-8 card-hover"
            >
              <h3 className="text-3xl font-bold mb-4">{service.title}</h3>
              <p className="text-muted mb-6">{service.description}</p>
              
              {service.code && (
                <div className="bg-background border border-accent/30 rounded-xl p-6 shadow-lg shadow-accent/10 overflow-hidden">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                      { label: "X-Ray", color: "bg-blue-100 text-blue-600", icon: "🫁" },
                      { label: "CT Scan", color: "bg-purple-100 text-purple-600", icon: "🧠" },
                      { label: "MRI", color: "bg-indigo-100 text-indigo-600", icon: "🔋" },
                      { label: "Ultrasound", color: "bg-pink-100 text-pink-600", icon: "📡" },
                    ].map((item, i) => (
                      <div key={i} className={`aspect-square rounded-xl ${item.color} flex flex-col items-center justify-center gap-2 hover:scale-105 transition-transform cursor-pointer`}>
                        <span className="text-4xl">{item.icon}</span>
                        <span className="text-xs font-bold">{item.label}</span>
                      </div>
                    ))}
                  </div>
                  <div className="mt-6 flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-100">
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                      <span className="text-sm font-semibold text-gray-600">Model Status:</span>
                      <span className="text-sm font-bold text-green-600">Active (98.4%)</span>
                    </div>
                    <div className="text-xs text-gray-400 font-mono">v2.4.0</div>
                  </div>
                </div>
              )}
              
              {service.tag && !service.code && (
                <div className="inline-block bg-accent/10 text-accent px-4 py-2 rounded-full text-sm font-semibold">
                  {service.tag}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="mt-16 bg-card border border-border rounded-2xl p-8">
          <div className="flex flex-wrap gap-4 justify-center">
            {features.map((feature, index) => (
              <div
                key={index}
                className="bg-background border border-border px-6 py-3 rounded-full text-sm font-medium hover:border-accent transition-colors"
              >
                {feature}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
