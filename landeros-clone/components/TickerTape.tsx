"use client";

export default function TickerTape() {
  const features = [
    "Data-Driven Decisions",
    "Personalized Experiences",
    "Automation",
    "Scalable Solutions",
    "Real-Time Insights",
    "Virtual Assistance",
    "Cost Effective",
    "Faster Innovation",
  ];

  // Duplicate for seamless loop
  const allFeatures = [...features, ...features];

  return (
    <div className="relative overflow-hidden bg-accent/5 border-y border-accent/20 py-6">
      <div className="flex animate-scroll">
        {allFeatures.map((feature, index) => (
          <div
            key={index}
            className="flex items-center gap-4 px-8 whitespace-nowrap"
          >
            <span className="text-lg font-semibold text-foreground">
              {feature}
            </span>
            <span className="text-accent">•</span>
          </div>
        ))}
      </div>
    </div>
  );
}
