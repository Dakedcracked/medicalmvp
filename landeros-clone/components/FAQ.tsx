"use client";

import { useState } from "react";

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const faqs = [
    {
      question: "What is included in the Starter plan?",
      answer: "The Starter plan includes basic AI tools usage, limited automation features, real-time reporting, and basic chatbot integration.",
    },
    {
      question: "Can I switch plans later?",
      answer: "Yes, you can upgrade or downgrade your plan at any time. Changes will be reflected in your next billing cycle.",
    },
    {
      question: "How secure is my data?",
      answer: "We use enterprise-grade encryption and follow industry best practices to ensure your data is always secure and protected.",
    },
    {
      question: "Can I integrate this platform with other tools?",
      answer: "Yes, we offer integrations with popular tools like Slack, Notion, Zapier, HubSpot, Salesforce, and many more.",
    },
    {
      question: "Do you offer a free trial?",
      answer: "Yes, all plans come with a 14-day free trial. No credit card required to get started.",
    },
    {
      question: "What payment methods do you accept?",
      answer: "We accept all major credit cards, PayPal, and wire transfers for enterprise customers.",
    },
    {
      question: "How does the 2% donation work?",
      answer: "We automatically donate 2% of your membership fees to pediatric wellbeing organizations at no extra cost to you.",
    },
    {
      question: "What makes your platform different?",
      answer: "Our platform combines cutting-edge AI technology with user-friendly design, backed by dedicated support and continuous innovation.",
    },
  ];

  return (
    <section className="py-24 px-6 bg-card/30">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="font-display text-5xl md:text-6xl font-bold mb-6">
            Questions answered
          </h2>
          <p className="text-lg text-muted mb-8">
            We're here to help you and solve objections. Find answers to the most common questions below.
          </p>
          <a
            href="#contact"
            className="inline-block bg-accent text-background px-8 py-3 rounded-full font-semibold hover:bg-accent/90 transition-all"
          >
            Contact Sales Now
          </a>
        </div>

        <div className="space-y-4">
          {faqs.map((faq, index) => (
            <div
              key={index}
              className="bg-card border border-border rounded-2xl overflow-hidden"
            >
              <button
                onClick={() => setOpenIndex(openIndex === index ? null : index)}
                className="w-full flex items-center justify-between p-6 text-left hover:bg-card/50 transition-colors"
              >
                <span className="font-semibold text-lg">{faq.question}</span>
                <svg
                  className={`w-6 h-6 text-accent transition-transform ${
                    openIndex === index ? "rotate-180" : ""
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>
              {openIndex === index && (
                <div className="px-6 pb-6">
                  <p className="text-muted leading-relaxed">{faq.answer}</p>
                </div>
              )}
            </div>
          ))}
        </div>

        <p className="text-center text-sm text-muted mt-12">
          Feel free to mail us for any enquiries:{" "}
          <a href="mailto:landeros@email.com" className="text-accent hover:underline">
            landeros@email.com
          </a>
        </p>
      </div>
    </section>
  );
}
