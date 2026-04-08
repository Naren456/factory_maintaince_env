import gradio as gr
import pandas as pd
import time
from server.factory_env_environment import FactoryEnvironment
from models import FactoryAction

# Initialize environment
env = FactoryEnvironment()
obs = env.reset()

def get_machine_data():
    data = []
    for m in env.machines:
        data.append({
            "ID": m.id,
            "Status": m.status.upper(),
            "Health": f"{m.health*100:.1f}%",
            "Last Maint": m.last_maint
        })
    return pd.DataFrame(data)

def update_ui():
    m_df = get_machine_data()
    metrics = {
        "Budget": f"${env.budget:.2f}",
        "Prod Rate": f"{obs.production_rate:.1f}%",
        "Step": f"{env.state.step_count}/{env.MAX_STEPS}"
    }
    return m_df, metrics["Budget"], metrics["Prod Rate"], metrics["Step"], obs.last_event

def handle_action(action_type, m_id=None):
    global obs
    action = FactoryAction(type=action_type, machine_id=m_id)
    obs = env.step(action)
    return update_ui()

def reset_env():
    global obs
    obs = env.reset()
    return update_ui()

with gr.Blocks(title="Factory Maintenance AI") as demo:
    gr.Markdown("# 🏭 Factory Maintenance Digital Twin")
    
    with gr.Row():
        with gr.Column(scale=1):
            budget_view = gr.Textbox(label="Available Budget", value=f"${env.budget:.2f}")
            prod_view = gr.Textbox(label="Production Rate", value="100%")
            step_view = gr.Textbox(label="Current Step", value="0/50")
            reset_btn = gr.Button("Reset Simulation", variant="stop")
            
        with gr.Column(scale=3):
            machine_table = gr.Dataframe(value=get_machine_data(), interactive=False)
            log_view = gr.Textbox(label="Event Log", value="Factory ready.")

    gr.Markdown("### Control Panel")
    with gr.Row():
        action_select = gr.Radio(["wait", "inspect", "repair", "replace"], label="Action Type", value="wait")
        machine_select = gr.Dropdown([0, 1, 2], label="Target Machine", value=0)
        submit_btn = gr.Button("Execute Action", variant="primary")

    # Event handlers
    submit_btn.click(
        handle_action, 
        inputs=[action_select, machine_select], 
        outputs=[machine_table, budget_view, prod_view, step_view, log_view]
    )
    
    reset_btn.click(
        reset_env,
        outputs=[machine_table, budget_view, prod_view, step_view, log_view]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
