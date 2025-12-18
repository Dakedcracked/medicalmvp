"use client";

import { useState } from "react";

export default function Pricing() {
  const [isYearly, setIsYearly] = useState(false);

  const plans = [
    {
      name: "Starter",
      price: 750,
      features: [
        "Basic AI tools usage",
        "Limited automation features",
        "Real-time reporting",
        "Basic chatbot integration",
      ],
    },
    {
      name: "Pro",
      price: 1700,
      popular: true,
      features: [
        "Everything in starter plan",
        "Integrations with 3rd-party",
        "Advanced analytics",
        "Premium chatbot features",
        "Cross-platform integrations",
      ],
    },
    {
      name: "Enterprise",
      price: 4700,
      features: [
        "Everything in pro plan",
        "Dedicated growth manager",
        "Custom reports & dashboards",
        "Fully customized AI solutions",
        "Scalable AI solutions",
      ],
    },
  ];

  const calculatePrice = (price: number) => {
    if (isYearly) {
      return Math.round(price * 0.7); // 30% off
    }
    return price;
  };

  return (
    <section id="pricing" className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="font-display text-5xl md:text-6xl font-bold mb-8">
            Pricing plans
          </h2>
          
          <div className="inline-flex items-center gap-4 bg-card border border-border rounded-full p-2">
            <button
              onClick={() => setIsYearly(false)}
              className={`px-6 py-2 rounded-full text-sm font-semibold transition-all ${
                !isYearly ? "bg-accent text-background" : "text-foreground"
              }`}
            >
              Monthly
            </button>
            <button
              onClick={() => setIsYearly(true)}
              className={`px-6 py-2 rounded-full text-sm font-semibold transition-all ${
                isYearly ? "bg-accent text-background" : "text-foreground"
              }`}
            >
              Yearly
            </button>
            {isYearly && (
              <span className="text-accent text-sm font-semibold pr-4">30% off</span>
            )}
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-8">
          {plans.map((plan, index) => (
            <div
              key={index}
              className={`bg-card border rounded-2xl p-8 card-hover relative ${
                plan.popular ? "border-accent" : "border-border"
              }`}
            >
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2 bg-accent text-background px-4 py-1 rounded-full text-xs font-bold">
                  Popular
                </div>
              )}
              
              <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
              <div className="mb-6">
                <span className="text-5xl font-bold">${calculatePrice(plan.price)}</span>
                <span className="text-muted">/month</span>
              </div>
              
              <a
                href="#"
                onClick={(e) => {
                  e.preventDefault();
                  console.log(`Get Started with ${plan.name} clicked`);
                }}
                className="block w-full bg-accent text-background text-center px-6 py-3 rounded-full font-semibold hover:bg-accent/90 transition-all mb-6"
              >
                Get Started for Free
              </a>
              
              <div className="space-y-3">
                <p className="text-sm font-semibold mb-4">
                  {plan.name} plan includes
                </p>
                {plan.features.map((feature, idx) => (
                  <div key={idx} className="flex items-start gap-2">
                    <svg
                      className="w-5 h-5 text-accent flex-shrink-0 mt-0.5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                    <span className="text-sm text-muted">{feature}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <p className="text-center text-sm text-muted">
          We donate 2% of your membership to pediatric wellbeing
        </p>
      </div>
    </section>
  );
}
