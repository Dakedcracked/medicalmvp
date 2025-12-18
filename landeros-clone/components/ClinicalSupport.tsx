"use client";

export default function ClinicalSupport() {
  const features = [
    {
      icon: "🎯",
      title: "Accurate Detection",
      description: "80.4% accuracy in identifying 9 different chest pathologies from X-ray images",
      stat: "80.4%",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: "⚡",
      title: "Instant Results",
      description: "Get AI-powered analysis in under 2 seconds, accelerating diagnosis workflow",
      stat: "<2s",
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: "🔒",
      title: "HIPAA Compliant",
      description: "Enterprise-grade security with full HIPAA compliance for patient data protection",
      stat: "100%",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: "📊",
      title: "Clinical Insights",
      description: "Detailed reports with confidence scores and region-specific findings",
      stat: "24/7",
      color: "from-orange-500 to-red-500"
    }
  ];

  return (
    <section className="py-24 px-6 bg-white">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16">
          <span className="inline-block bg-accent/10 text-accent px-4 py-2 rounded-full text-sm font-semibold mb-4">
            🏥 CLINICAL DECISION SUPPORT
          </span>
          <h2 className="text-5xl md:text-6xl font-bold mb-6">
            AI-Powered <span className="text-accent">Diagnostic</span> Assistance
          </h2>
          <p className="text-xl text-muted max-w-3xl mx-auto">
            Enhance clinical decision-making with AI that detects pathologies, reduces diagnostic time, and improves patient outcomes
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="group bg-white rounded-3xl p-8 border border-gray-200 hover:border-accent hover:shadow-2xl transition-all duration-300 cursor-pointer"
            >
              {/* Icon */}
              <div className="text-6xl mb-4 group-hover:scale-110 transition-transform">
                {feature.icon}
              </div>

              {/* Stat */}
              <div className={`text-4xl font-bold mb-3 bg-gradient-to-r ${feature.color} bg-clip-text text-transparent`}>
                {feature.stat}
              </div>

              {/* Title */}
              <h3 className="text-xl font-bold mb-3 group-hover:text-accent transition-colors">
                {feature.title}
              </h3>

              {/* Description */}
              <p className="text-muted text-sm leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {/* Pathologies Detection */}
        <div className="bg-gradient-to-br from-gray-50 to-white rounded-3xl p-12 border border-gray-200">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold mb-4">Detectable Conditions</h3>
            <p className="text-muted">Our AI can identify these pathologies from chest X-rays</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {[
              { name: "Pneumonia", icon: "🫁", color: "bg-red-100 text-red-700 border-red-200" },
              { name: "Cardiomegaly", icon: "❤️", color: "bg-pink-100 text-pink-700 border-pink-200" },
              { name: "Effusion", icon: "💧", color: "bg-blue-100 text-blue-700 border-blue-200" },
              { name: "Atelectasis", icon: "🌬️", color: "bg-cyan-100 text-cyan-700 border-cyan-200" },
              { name: "Mass", icon: "⚫", color: "bg-gray-100 text-gray-700 border-gray-200" },
              { name: "Nodule", icon: "🔴", color: "bg-orange-100 text-orange-700 border-orange-200" },
              { name: "Pneumothorax", icon: "💨", color: "bg-green-100 text-green-700 border-green-200" },
              { name: "Infiltration", icon: "🦠", color: "bg-purple-100 text-purple-700 border-purple-200" },
              { name: "No Finding", icon: "✅", color: "bg-emerald-100 text-emerald-700 border-emerald-200" }
            ].map((condition, index) => (
              <div 
                key={index}
                className={`${condition.color} rounded-2xl p-4 border-2 text-center hover:scale-105 transition-transform cursor-pointer`}
              >
                <div className="text-3xl mb-2">{condition.icon}</div>
                <div className="text-sm font-semibold">{condition.name}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="mt-16 text-center">
          <div className="inline-flex flex-col sm:flex-row gap-4">
            <a 
              href="/analyze" 
              className="inline-flex items-center justify-center gap-2 bg-accent text-white px-8 py-4 rounded-full font-semibold hover:bg-accent-dark transition-all hover:scale-105 shadow-lg shadow-accent/30"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              Analyze Medical Image
            </a>
            <a 
              href="/blog" 
              className="inline-flex items-center justify-center gap-2 bg-white text-accent border-2 border-accent px-8 py-4 rounded-full font-semibold hover:bg-accent hover:text-white transition-all hover:scale-105"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              Read Research
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
