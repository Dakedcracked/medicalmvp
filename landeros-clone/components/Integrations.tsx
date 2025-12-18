"use client";

export default function Integrations() {
  const integrations = [
    "Slack", "Notion", "Zapier", "HubSpot", "Salesforce", 
    "Google Workspace", "Microsoft Teams", "Asana"
  ];

  return (
    <section className="py-24 px-6">
      <div className="max-w-7xl mx-auto text-center">
        <p className="text-sm text-accent font-semibold tracking-wider mb-4">INTEGRATIONS</p>
        <h2 className="font-display text-5xl md:text-6xl font-bold mb-12">
          Seamless AI Integrations That Work Everywhere
        </h2>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {integrations.map((integration, index) => (
            <div
              key={index}
              className="bg-card border border-border rounded-2xl p-8 card-hover flex items-center justify-center"
            >
              <span className="text-lg font-semibold">{integration}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
