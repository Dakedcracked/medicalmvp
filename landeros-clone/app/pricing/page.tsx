"use client";

import { useState } from "react";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Pricing from "@/components/Pricing";

export default function PricingPage() {
  return (
    <main className="min-h-screen relative">
      <div className="relative z-10">
        <Navbar />
        
        <div className="pt-32 pb-20 px-6">
          <div className="max-w-7xl mx-auto">
            {/* Header */}
            <div className="text-center mb-16">
              <h1 className="text-6xl font-bold mb-6">
                Simple, <span className="text-accent">Transparent</span> Pricing
              </h1>
              <p className="text-xl text-muted max-w-3xl mx-auto">
                Choose the perfect plan for your healthcare facility. All plans include our core AI features with clinical-grade accuracy.
              </p>
            </div>

            {/* Pricing Component */}
            <Pricing />

            {/* Features Comparison */}
            <div className="mt-20 bg-white rounded-3xl p-12 border border-border">
              <h2 className="text-4xl font-bold mb-12 text-center">Feature Comparison</h2>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-4 px-6 font-bold">Features</th>
                      <th className="text-center py-4 px-6 font-bold">Starter</th>
                      <th className="text-center py-4 px-6 font-bold">Professional</th>
                      <th className="text-center py-4 px-6 font-bold">Enterprise</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border">
                      <td className="py-4 px-6">Monthly Scans</td>
                      <td className="text-center py-4 px-6 text-muted">100</td>
                      <td className="text-center py-4 px-6 text-muted">Unlimited</td>
                      <td className="text-center py-4 px-6 text-muted">Unlimited</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-4 px-6">AI Accuracy</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-4 px-6">8 Pathology Detection</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-4 px-6">API Access</td>
                      <td className="text-center py-4 px-6 text-muted">-</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-4 px-6">Priority Support</td>
                      <td className="text-center py-4 px-6 text-muted">-</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-4 px-6">Custom Integration</td>
                      <td className="text-center py-4 px-6 text-muted">-</td>
                      <td className="text-center py-4 px-6 text-muted">-</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                    </tr>
                    <tr className="border-b border-border">
                      <td className="py-4 px-6">Dedicated Account Manager</td>
                      <td className="text-center py-4 px-6 text-muted">-</td>
                      <td className="text-center py-4 px-6 text-muted">-</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                    </tr>
                    <tr>
                      <td className="py-4 px-6">On-Premise Deployment</td>
                      <td className="text-center py-4 px-6 text-muted">-</td>
                      <td className="text-center py-4 px-6 text-muted">-</td>
                      <td className="text-center py-4 px-6 text-accent">✓</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* FAQ */}
            <div className="mt-20">
              <h2 className="text-4xl font-bold mb-12 text-center">Frequently Asked Questions</h2>
              
              <div className="grid md:grid-cols-2 gap-8">
                <div className="bg-white rounded-2xl p-8 border border-border">
                  <h3 className="text-xl font-bold mb-3">Can I upgrade my plan later?</h3>
                  <p className="text-muted">Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately.</p>
                </div>

                <div className="bg-white rounded-2xl p-8 border border-border">
                  <h3 className="text-xl font-bold mb-3">Is there a free trial?</h3>
                  <p className="text-muted">Yes, all plans come with a 14-day free trial. No credit card required.</p>
                </div>

                <div className="bg-white rounded-2xl p-8 border border-border">
                  <h3 className="text-xl font-bold mb-3">What payment methods do you accept?</h3>
                  <p className="text-muted">We accept all major credit cards, wire transfers, and purchase orders for enterprise customers.</p>
                </div>

                <div className="bg-white rounded-2xl p-8 border border-border">
                  <h3 className="text-xl font-bold mb-3">Is my data secure?</h3>
                  <p className="text-muted">Yes, we are HIPAA compliant and use enterprise-grade encryption for all medical data.</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <Footer />
      </div>
    </main>
  );
}
