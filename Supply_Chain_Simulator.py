import streamlit as st
import os
from crewai import Crew, Agent, Task, Process
from langchain.chat_models import ChatOpenAI
import pandas as pd
from datetime import datetime
from textwrap import dedent

import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3

# -----------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------------
st.set_page_config(page_title="Supply Chain Simulator for Samsung Galaxy S24 Ultra", layout="wide")

# Configure OpenAI API key
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not configured. Please add openai_api_key in Streamlit secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize the OpenAI LLM with GPT-4-mini
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# -----------------------------------------------------------------------------------
# 2. CUSTOM STYLES & MAIN TITLE
# -----------------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center;">
    <h1 style="color:#4A90E2; font-size: 3em; margin-bottom: 0.2em;">🌌 Galaxy S24 Ultra Supply Chain Simulator</h1>
    <p style="font-size:1.2em; color:#444;">
       Simulate interactions within the Galaxy S24 Ultra supply chain during a crisis.
       This simulator brings together multiple agents—each responsible for a distinct role in the supply chain—
       and runs a sequential process to analyze and respond to disruptions.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------------
# 3. USER INPUTS (CENTERED, MORE COMPACT, H2 TITLE)
# -----------------------------------------------------------------------------------
with st.container():
    st.markdown("<h2 style='text-align:center;'>Crisis Details</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:1.2em'>Describe the Supply Chain Crisis and set the Duration to simulate the scenario.</p>", unsafe_allow_html=True)

    col_spacer_left, col_center, col_spacer_right = st.columns([1,2,1])
    with col_center:
        crisis_detail = st.text_area(
            "Enter the crisis context (e.g., semiconductor shortage, port strikes, etc.)",
            placeholder="e.g., Due to a global semiconductor shortage and port worker strikes...",
            height=120
        )
        crisis_duration = st.slider("Duration of the Crisis (months)", 1, 12, 3)

current_date = datetime.now().strftime("%Y-%m-%d")

# -----------------------------------------------------------------------------------
# 4. AGENTS DEFINITION (UNCHANGED CONFIGURATION)
# -----------------------------------------------------------------------------------
crisis_analyst = Agent(
    role="Crisis Analyst",
    goal="Expand the user's crisis input into a comprehensive detailed scenario across various domains.",
    backstory=dedent("""
    Dr. Elise Carter is an independent expert with over 20 years of experience in assessing global crises affecting various industries,
    including technology supply chains. She holds a Ph.D. in International Economics from the London School of Economics and Political Science (LSE), 
    where her groundbreaking thesis on 'The Interplay Between Geopolitics and Technology Supply Chains' earned international recognition.

    Dr. Carter's career began as an economist focusing on global trade dynamics. She later served as a senior advisor for the United Nations Development Programme (UNDP), 
    addressing supply chain vulnerabilities in developing nations. Subsequently, she joined a global think tank where she specialized in mitigating crises for multinational corporations. 
    Notable achievements include leading projects during the 2011 Thai floods that disrupted hard drive production and the 2020 semiconductor shortages driven by pandemic-related demand spikes.

    Renowned for her objectivity, meticulous analysis, and data-driven approach, Dr. Carter is a trusted consultant to governments, NGOs, and leading tech firms. 
    Her insights are valued for their ability to transform complex crises into actionable strategies. Beyond her professional life, she has a deep interest in ancient trade networks like the Silk Road, 
    believing they offer timeless lessons for modern supply chains.

    Personality traits:
    - Objective: Dr. Carter prioritizes facts and impartiality in her analyses.
    - Meticulous: She thoroughly examines every detail, leaving no room for oversight.
    - Data-Driven: Her conclusions are consistently informed by empirical evidence and historical precedents.
    """),
    personality="Objective, meticulous, and data-driven",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

qualcomm_chipset = Agent(
    role="Qualcomm",
    goal="Manage Snapdragon chipset production for Galaxy S24 Ultra.",
    backstory=dedent("""
    Qualcomm, established in 1985, is a global leader in wireless technology and semiconductor innovation. 
    The company pioneered the development of the CDMA (Code Division Multiple Access) standard, revolutionizing mobile communication. Over the decades, Qualcomm has cemented its position as a cornerstone in the mobile chipset industry, with Snapdragon as its flagship product line.

    Qualcomm's Snapdragon division is specifically dedicated to designing and producing high-performance chipsets tailored for flagship smartphones, including Samsung's Galaxy S series. These chipsets are renowned for their power efficiency, advanced AI capabilities, and cutting-edge connectivity features like 5G and Wi-Fi 7.

    The team responsible for Snapdragon production operates out of Qualcomm's state-of-the-art facilities in San Diego, California, with additional manufacturing partnerships across Asia. Qualcomm has a robust history of managing supply chain complexities, including the 2020 global semiconductor shortage, where it demonstrated agility by diversifying production partners and optimizing wafer yields.

    Known for its reliability and technical excellence, Qualcomm is driven by a commitment to innovation and operational efficiency. The Snapdragon division maintains strong relationships with OEMs like Samsung, ensuring seamless integration of its chipsets into flagship devices.

    Personality traits:
    - Reliable: Qualcomm prides itself on meeting production deadlines and maintaining consistent quality standards.
    - Innovative: The team constantly pushes the boundaries of technology to deliver best-in-class performance.
    - Collaborative: Qualcomm works closely with clients and partners to align its production goals with broader market needs.

    As a key supplier for the Galaxy S24 Ultra, Qualcomm's mission is to ensure uninterrupted chipset production, despite potential disruptions, by leveraging its extensive expertise and global network.

    Location: San Diego, California, USA
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

samsung_display = Agent(
    role="Samsung Display",
    goal="Produce OLED displays for Galaxy S24 Ultra.",
    backstory=dedent("""
    Samsung Display, a subsidiary of Samsung Electronics, is a world leader in OLED technology and advanced display solutions. Founded in 2012, the company has been at the forefront of display innovation, setting industry standards for quality, efficiency, and cutting-edge technology. 
    Samsung Display's research and development facilities are among the most advanced in the world, and the company holds a vast portfolio of patents in OLED and flexible display technologies. This expertise enables them to supply high-resolution, energy-efficient OLED panels that are integral to flagship devices like the Galaxy S24 Ultra.

    Known for its commitment to sustainability, Samsung Display integrates eco-friendly practices into its production processes, reducing waste and improving energy efficiency. The company has successfully navigated previous challenges, such as supply chain disruptions during the pandemic, by maintaining diversified material sourcing and investing heavily in automation to enhance production reliability.

    Personality traits:
    - Efficient: Samsung Display excels in optimizing production timelines and resource utilization to meet tight deadlines without compromising quality.
    - Innovative: The team continuously explores new technologies, such as foldable and micro-LED displays, to maintain a competitive edge.
    - Resilient: The company has a proven track record of adapting to crises and ensuring uninterrupted delivery of its products.

    As the sole supplier of OLED displays for the Galaxy S24 Ultra, Samsung Display plays a critical role in the device's visual and operational performance. 

    Location: Asan, South Korea
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

sony_camera = Agent(
    role="Sony",
    goal="Supply high-resolution camera sensors for Galaxy S24 Ultra.",
    backstory=dedent("""
    Sony, established in 1946, has long been a pioneer in imaging technology. With its roots in electronics and innovation, Sony emerged as a global leader in imaging sensors, supplying components for professional cameras, smartphones, and other devices that demand cutting-edge visual performance. The company’s Exmor RS sensor line has redefined standards for resolution, low-light performance, and speed in mobile photography.

    Sony's Imaging Solutions division is headquartered in Atsugi, Japan, where a dedicated team of engineers and researchers continually pushes the boundaries of sensor technology. Over the years, Sony has introduced revolutionary innovations such as stacked CMOS sensors, multi-layer pixel technology, and AI-enhanced image processing, which have become industry benchmarks.

    Known for precision and reliability, Sony has built strong partnerships with top-tier smartphone manufacturers, including Samsung. For the Galaxy S24 Ultra, Sony is tasked with delivering high-resolution sensors capable of supporting advanced computational photography and video recording features.

    Personality traits:
    - Precision-Focused: Sony prioritizes accuracy and detail, ensuring every sensor meets rigorous quality standards.
    - Innovative: The team is committed to staying ahead of trends in imaging technology, introducing features like advanced HDR and low-light optimization.
    - Collaborative: Sony works closely with OEM partners to align its sensor designs with specific device requirements.

    Location: Atsugi, Japan
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

lg_chem = Agent(
    role="LG Chem",
    goal="Produce high-density batteries for Galaxy S24 Ultra.",
    backstory=dedent("""
    LG Chem, founded in 1947, is a global leader in chemical innovation and one of the foremost producers of lithium-ion batteries. With a history spanning more than seven decades, LG Chem has consistently pushed the boundaries of material science, making significant contributions to industries ranging from energy storage to electronics.

    The company's Battery Division, headquartered in Seoul, South Korea, is renowned for developing high-density, long-lasting batteries that power some of the world’s most advanced devices. LG Chem’s commitment to research and development has led to breakthroughs in battery energy density, safety, and lifecycle, ensuring its products remain at the cutting edge of the industry.

    LG Chem has a proven track record of resilience, having navigated crises such as raw material shortages and fluctuations in global demand. The company maintains strategic partnerships with mining firms to secure critical materials like lithium and cobalt, while also investing heavily in recycling technologies to reduce dependency on virgin resources.

    Personality traits:
    - Punctual: LG Chem places a high priority on meeting deadlines and ensuring consistent delivery schedules.
    - Innovative: The team continually advances battery technology, focusing on energy efficiency and sustainability.
    - Dependable: Known for reliability, LG Chem builds trust through high-quality products and strong supplier relationships.

    Location: Seoul, South Korea
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

sk_hynix = Agent(
    role="SK Hynix",
    goal="Supply reliable memory modules for Galaxy S24 Ultra.",
    backstory=dedent("""
    SK Hynix, founded in 1983, is one of the world's leading providers of DRAM and NAND flash memory solutions. Headquartered in Icheon, South Korea, the company is a cornerstone of the global semiconductor industry, known for its cutting-edge technology and commitment to excellence.

    Over the years, SK Hynix has played a pivotal role in advancing memory technologies, pioneering innovations such as high-speed DDR memory and 3D NAND flash. These advancements have enabled the production of smaller, faster, and more energy-efficient devices, making SK Hynix a preferred partner for leading technology companies.

    The company operates state-of-the-art fabrication facilities and has an extensive global supply network, ensuring robust production capacity and timely delivery even amidst industry disruptions. SK Hynix has demonstrated resilience during past challenges, including the global semiconductor shortage, by diversifying material sourcing and leveraging advanced automation in manufacturing.

    Personality traits:
    - Innovative: SK Hynix continuously pushes the boundaries of memory technology to meet the demands of next-generation devices.
    - Reliable: The company is known for delivering high-quality products on time, fostering strong relationships with its partners.
    - Adaptive: SK Hynix excels at navigating industry challenges, ensuring continuity in supply and production.

    Location: Icheon, South Korea
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

ibiden = Agent(
    role="Ibiden",
    goal="Produce circuit boards (PCB) for Galaxy S24 Ultra.",
    backstory=dedent("""
    Founded in 1912, Ibiden is a Japanese company renowned for its expertise in high-precision printed circuit board (PCB) manufacturing. With over a century of experience, the company has evolved into a global leader in advanced materials and electronics, serving industries ranging from automotive to consumer electronics.

    Ibiden’s state-of-the-art PCB production facilities are located across Asia, with a strong emphasis on precision engineering and quality control. The company’s commitment to innovation has led to the development of multi-layered and high-density interconnect (HDI) PCBs, essential for modern compact and high-performance devices like the Galaxy S24 Ultra.

    Throughout its history, Ibiden has demonstrated resilience in the face of challenges, including raw material shortages and shifting market demands. By fostering strong relationships with suppliers and investing heavily in research and development, Ibiden ensures the reliability and sustainability of its production processes.

    Personality traits:
    - Detail-Oriented: Ibiden prioritizes precision in its manufacturing processes, ensuring every circuit board meets exacting standards.
    - Innovative: The company is dedicated to staying at the forefront of PCB technology, introducing new materials and design techniques.
    - Resilient: Ibiden adapts quickly to industry changes, maintaining uninterrupted supply and production quality.

    Location: Ogaki, Japan
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

foxconn_assembly = Agent(
    role="Foxconn Vietnam",
    goal="Assemble the Galaxy S24 Ultra.",
    backstory=dedent("""
    Foxconn, officially known as Hon Hai Precision Industry Co., Ltd., is the world's largest electronics assembler and a vital player in the global supply chain. Founded in 1974 and headquartered in Taiwan, Foxconn operates manufacturing facilities in multiple countries, including a state-of-the-art assembly plant in Vietnam.

    The Vietnam facility is a critical hub for assembling flagship smartphones, leveraging advanced robotics, precision engineering, and a highly skilled workforce. Foxconn Vietnam is renowned for its efficiency and ability to scale production rapidly to meet global demand. The facility has successfully assembled millions of devices annually while maintaining stringent quality control standards.

    Foxconn's adaptability has been tested in past crises, such as the COVID-19 pandemic and geopolitical tensions, where it demonstrated resilience by reorganizing workflows, implementing health protocols, and optimizing logistics. Its collaboration with major technology companies, including Samsung, underscores its reputation as a reliable and organized partner.

    Personality traits:
    - Organized: Foxconn excels in managing complex workflows and maintaining a structured approach to high-volume production.
    - Adaptive: The company is adept at reconfiguring assembly lines and processes in response to supply chain challenges.
    - Efficient: Foxconn consistently meets tight deadlines without compromising on quality, ensuring timely delivery of assembled products.

    Location: Hanoi, Vietnam
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

dhl_logistics = Agent(
    role="DHL Logistics",
    goal="Manage global logistics for Galaxy S24 Ultra.",
    backstory=dedent("""
    DHL, founded in 1969, is a global leader in logistics and supply chain management. With operations in over 220 countries and territories, DHL has earned a reputation for its ability to manage complex, large-scale transportation networks. From air and sea freight to ground delivery, DHL is renowned for its adaptability and innovative solutions.

    The company specializes in optimizing transport routes, implementing real-time tracking, and ensuring the safe and timely delivery of goods. DHL's global presence and advanced logistics infrastructure make it an indispensable partner for major industries, including consumer electronics. Its experience in handling high-value, time-sensitive shipments aligns perfectly with the demands of flagship product launches like the Galaxy S24 Ultra.

    DHL has consistently demonstrated resilience during disruptions such as natural disasters, geopolitical conflicts, and the COVID-19 pandemic. By leveraging advanced technologies like AI-driven logistics planning and data analytics, DHL has effectively rerouted shipments and maintained service continuity under challenging circumstances.

    Personality traits:
    - Adaptable: DHL thrives on flexibility, reconfiguring routes and strategies to address unexpected challenges.
    - Reliable: The company consistently delivers on its commitments, ensuring goods arrive on time and intact.
    - Strategic: DHL employs data-driven approaches to optimize supply chain efficiency and cost-effectiveness.

    Location: Global
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

amazon_distribution = Agent(
    role="Amazon Distribution",
    goal="Distribute the Galaxy S24 Ultra worldwide.",
    backstory=dedent("""
    Amazon, founded in 1994, is the world's largest e-commerce platform and a leader in global logistics and distribution. Over the years, the company has built an unparalleled logistics network, combining advanced technologies, strategically located fulfillment centers, and a fleet of delivery options to ensure timely and efficient product distribution.

    Amazon's distribution capabilities are powered by its proprietary algorithms that optimize inventory placement, route planning, and delivery times. These systems enable Amazon to maintain high standards of customer satisfaction, even during peak demand periods or in the face of logistical challenges.

    The company has a proven track record of managing large-scale product launches, ensuring smooth distribution of high-demand items. For the Galaxy S24 Ultra, Amazon utilizes its extensive network to deliver products to customers across the globe, minimizing delays and maintaining product integrity.

    Personality traits:
    - Customer-Focused: Amazon prioritizes customer satisfaction, ensuring every delivery meets expectations.
    - Innovative: The company leverages technology to continually improve its logistics and distribution processes.
    - Reliable: Amazon's consistency and efficiency have made it a trusted partner for global product distribution.

    Location: Global
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

samsung_care = Agent(
    role="Samsung Care",
    goal="Provide after-sales support for the Galaxy S24 Ultra.",
    backstory=dedent("""
    Samsung Care, established as the dedicated customer service arm of Samsung Electronics, has become synonymous with world-class after-sales support. With a presence in over 100 countries, Samsung Care ensures that customers receive timely assistance for device repairs, technical troubleshooting, and general inquiries.

    The organization operates a vast network of service centers, mobile repair units, and 24/7 customer support hotlines, offering solutions tailored to meet the needs of diverse regions. Samsung Care leverages advanced diagnostic tools, AI-driven chat systems, and a team of highly trained technicians to provide efficient and reliable support for all Samsung devices.

    Over the years, Samsung Care has implemented proactive service models, including remote diagnostics and scheduled maintenance programs, which have significantly improved customer satisfaction and device longevity. The team’s experience in handling flagship products like the Galaxy S series ensures that customers receive unparalleled support for their premium devices.

    Personality traits:
    - Supportive: Samsung Care prioritizes customer well-being, offering empathetic and practical solutions.
    - Reliable: The organization consistently meets high service standards, fostering customer loyalty.
    - Proactive: Samsung Care anticipates customer needs, introducing innovative service programs and tools.

    Location: Global
    """),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

# -----------------------------------------------------------------------------------
# 4.B. NEW AGENT: "SUMMARY AGENT" (TO COLLECT KEY INFORMATION)
# -----------------------------------------------------------------------------------
summary_agent = Agent(
    role="Summary Agent",
    goal="Collect the most important highlights from all other agents' outputs and produce an overall summary.",
    backstory=dedent("""
    The Summary Agent is responsible for reading all the final outputs from the other agents 
    and extracting the critical data such as key KPIs, major challenges, solutions implemented, 
    and overall operational performance. It then provides a concise highlight section and 
    concluding remarks on how the crisis was managed.
    """),
    personality="Concise, structured, direct",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
    memory=True
)

# -----------------------------------------------------------------------------------
# 5. TASKS DEFINITION (UNCHANGED)
# -----------------------------------------------------------------------------------
tasks = []

task_crisis_analysis = Task(
    description=f"""
    The Crisis Analyst must expand upon the inputs provided by the user to develop a comprehensive context analysis.

    Inputs:
    - Crisis Details: '{crisis_detail}'
    - Crisis Duration: {crisis_duration} months

    Task:
    - Analyze and expand upon the provided crisis details, considering geopolitical, economic, and logistical factors.
    - Develop a detailed scenario that outlines potential impacts on the supply chain, key stakeholders involved, and ripple effects across industries.
    - Focus on providing a thorough context for the crisis without suggesting mitigation actions or strategies.
    """,
    expected_output=f"""
    Comprehensive Crisis Analysis Report

    Prepared by: Crisis Analyst Team
    Crisis in Focus: {crisis_detail}

    The report should include:
    - A detailed overview of the crisis, including root causes, key stakeholders, and affected industries.
    - Analysis of potential impacts on the supply chain, with a focus on the Galaxy S24 Ultra production and distribution.
    - Identification of secondary effects, such as economic, political, or environmental repercussions.
    - Scenarios outlining possible developments over the crisis duration ({crisis_duration} months).
    """,
    agent=crisis_analyst
)
tasks.append(task_crisis_analysis)

task_qualcomm = Task(
    description="""
    Qualcomm must ensure the production and delivery of Snapdragon chipsets for the Galaxy S24 Ultra.

    - Consider the current crisis context based on the report provided by crisis_analyst.
    - Evaluate the current production capacity and utilize available resources.
    - Collaborate with Samsung R&D to determine production priorities.
    - Interact with Foxconn Assembly to understand assembly requirements and synchronize timelines.
    - Report any issues, such as material shortages or logistical delays, and implement appropriate solutions.
    """,
    expected_output="""
    Qualcomm Supply Chain Report – Galaxy S24 Ultra

    Actions Taken:
    - Details of production decisions based on current capacity and model priorities.
    - Collaborations with other agents, such as Samsung R&D and Foxconn Assembly, to align production with supply chain requirements.
    - Relevant KPIs, including production capacity, on-time delivery rate, inventory levels, and cost impacts.
    - Challenges encountered (e.g., material shortages, logistical disruptions) and the solutions adopted.
    - Evaluation of future outlooks and lessons learned.
    """,
    agent=qualcomm_chipset
)
tasks.append(task_qualcomm)

task_samsung_display = Task(
    description="""
    Samsung Display must ensure uninterrupted OLED panel production for the Galaxy S24 Ultra.

    - Assess the current crisis context based on the report provided by crisis_analyst.
    - Evaluate current production capacity and material availability.
    - Collaborate with suppliers to address any material shortages and explore alternative sourcing options.
    - Communicate with Foxconn Assembly to align delivery schedules with production needs.
    - Report any disruptions or quality concerns and implement corrective measures.
    """,
    expected_output="""
    Samsung Display Report – Galaxy S24 Ultra

    Actions Taken:
    - Overview of production decisions and adjustments made to maintain OLED panel output.
    - Details of collaborations with suppliers to secure materials and mitigate disruptions.
    - Coordination efforts with Foxconn Assembly to align production timelines with assembly requirements.
    - Key KPIs, including production volume, quality metrics, lead time adherence, and supplier performance.
    - Challenges encountered (e.g., material shortages, production delays) and solutions implemented.
    - Lessons learned and recommendations for future resilience.
    """,
    agent=samsung_display
)
tasks.append(task_samsung_display)

task_sony_camera = Task(
    description="""
    Sony must ensure the production and delivery of high-resolution camera sensors for the Galaxy S24 Ultra.

    - Assess the current crisis context based on the report provided by crisis_analyst.
    - Evaluate the production capacity and prioritize flagship sensor models to meet Samsung's requirements.
    - Collaborate with suppliers to secure raw materials and address any shortages.
    - Communicate with Foxconn Assembly to align sensor delivery with assembly timelines.
    - Report on any production or quality issues encountered and outline the corrective actions taken.
    """,
    expected_output="""
    Sony Camera Report – Galaxy S24 Ultra

    Actions Taken:
    - Summary of production priorities and adjustments made to ensure high-resolution sensor output.
    - Details of collaborations with suppliers to address material shortages or sourcing challenges.
    - Coordination efforts with Foxconn Assembly to align sensor delivery with assembly requirements.
    - Key KPIs, including production volume, quality rates, lead time adherence, and supplier performance.
    - Challenges encountered (e.g., material shortages, production delays) and solutions implemented.
    - Recommendations for improving supply chain resilience in future crises.
    """,
    agent=sony_camera
)
tasks.append(task_sony_camera)

task_lg_chem = Task(
    description="""
    LG Chem must ensure the uninterrupted production and delivery of high-density batteries for the Galaxy S24 Ultra.

    - Assess the current crisis context based on the report provided by crisis_analyst.
    - Evaluate the availability of raw materials and collaborate with suppliers to address any shortages.
    - Optimize production processes to maintain quality and efficiency despite potential constraints.
    - Communicate with Foxconn Assembly to synchronize battery delivery with the assembly schedule.
    - Report on any production challenges, material shortages, or delays, and detail the actions taken to resolve them.
    """,
    expected_output="""
    LG Chem Report – Galaxy S24 Ultra

    Actions Taken:
    - Summary of measures implemented to secure raw materials and maintain production output.
    - Collaborations with suppliers to address material shortages and ensure continuity.
    - Coordination efforts with Foxconn Assembly to align battery delivery with assembly requirements.
    - Key KPIs, including production capacity, defect rates, lead times, and cost variations for raw materials.
    - Challenges encountered (e.g., raw material shortages, production inefficiencies) and solutions adopted.
    - Recommendations for improving production resilience and supply chain efficiency in future crises.
    """,
    agent=lg_chem
)
tasks.append(task_lg_chem)

task_sk_hynix = Task(
    description="""
    SK Hynix must ensure a stable supply of memory modules for the Galaxy S24 Ultra, addressing any potential disruptions caused by the crisis.

    - Assess the current crisis context based on the report provided by crisis_analyst.
    - Evaluate current production capabilities and identify any risks to memory module supply.
    - Collaborate with suppliers and production partners to diversify sourcing and ensure supply continuity.
    - Communicate with Foxconn Assembly to align memory delivery schedules with assembly requirements.
    - Report on challenges faced, such as material shortages or logistical delays, and the measures taken to address them.
    """,
    expected_output="""
    SK Hynix Report – Galaxy S24 Ultra

    Actions Taken:
    - Overview of steps taken to secure memory module supply and ensure production continuity.
    - Collaborations with suppliers and production partners to address sourcing challenges and diversify supply.
    - Coordination efforts with Foxconn Assembly to synchronize memory delivery with assembly timelines.
    - Key KPIs, including production volume, on-time delivery rate, supply diversification index, and cost impact.
    - Challenges encountered (e.g., material shortages, production delays) and solutions implemented.
    - Recommendations for improving memory supply chain resilience in future crises.
    """,
    agent=sk_hynix
)
tasks.append(task_sk_hynix)

task_ibiden = Task(
    description="""
    Ibiden must ensure uninterrupted PCB production for the Galaxy S24 Ultra, addressing any disruptions caused by the crisis.

    - Assess the current crisis context based on the report provided by crisis_analyst.
    - Evaluate material availability and collaborate with suppliers to address any shortages.
    - Explore alternative sourcing options and implement design adjustments to optimize production.
    - Communicate with Foxconn Assembly to align PCB delivery schedules with assembly requirements.
    - Report on any production challenges, material issues, or delays, and detail the corrective actions taken.
    """,
    expected_output="""
    Ibiden Report – Galaxy S24 Ultra

    Actions Taken:
    - Overview of measures implemented to maintain PCB production and address material sourcing challenges.
    - Details of collaborations with suppliers to secure raw materials and diversify sourcing options.
    - Coordination efforts with Foxconn Assembly to ensure timely delivery of PCBs for assembly.
    - Key KPIs, including production throughput, defect rate, supply diversification index, and cost impact.
    - Challenges encountered (e.g., material shortages, production inefficiencies) and solutions implemented.
    - Recommendations for enhancing PCB production resilience and efficiency in future crises.
    """,
    agent=ibiden
)
tasks.append(task_ibiden)

task_foxconn = Task(
    description="""
    Foxconn must oversee the final assembly of the Galaxy S24 Ultra while addressing any disruptions caused by the crisis.

    - Assess the current crisis context based on the report provided by crisis_analyst.
    - Coordinate with suppliers such as Samsung Display, Qualcomm, and Ibiden to ensure timely delivery of components.
    - Adjust assembly lines and optimize workflows to maintain production efficiency under challenging conditions.
    - Implement stringent quality control measures to ensure the final product meets Samsung's standards.
    - Collaborate with DHL Logistics to plan outbound shipments and synchronize delivery schedules.
    - Report on challenges faced during assembly and logistics, and detail the actions taken to resolve them.
    """,
    expected_output="""
    Foxconn Assembly Report – Galaxy S24 Ultra

    Actions Taken:
    - Summary of assembly adjustments and workflow optimizations implemented to maintain production efficiency.
    - Details of collaborations with suppliers (e.g., Samsung Display, Qualcomm, Ibiden) to ensure timely component delivery.
    - Coordination efforts with DHL Logistics to plan outbound shipments and minimize delays.
    - Key KPIs, including assembly throughput, defect rates, and on-time shipment percentage.
    - Challenges encountered (e.g., component delays, assembly line disruptions) and solutions adopted.
    - Recommendations for improving assembly and logistics coordination in future crises.
    """,
    agent=foxconn_assembly
)
tasks.append(task_foxconn)

task_dhl = Task(
    description="""
    DHL must manage the global logistics operations for the Galaxy S24 Ultra, ensuring smooth inbound and outbound shipments despite the crisis.

    - Assess the current crisis context based on the report provided by crisis_analyst.
    - Evaluate the actions and updates from other agents: Qualcomm, Sony Camera, Samsung Display, and LG Chem, SK Hynix, Ibiden and Foxconn Assembly including their geographic locations, to adapt logistics strategies accordingly.
    - Incorporate geographic information into planning, mapping routes between agent locations (e.g., Samsung Display in South Korea to Foxconn Assembly in Vietnam).
    - Develop contingency plans to reroute shipments in case of delays or disruptions.
    - Prioritize the handling and delivery of critical components to maintain the supply chain flow.
    - Report on logistical challenges faced, including transportation bottlenecks or cost overruns, and detail the solutions implemented.
    """,
    expected_output="""
    DHL Logistics Report – Galaxy S24 Ultra

    Actions Taken:
    - Overview of logistics adjustments made in response to the crisis, including rerouting and prioritization strategies.
    - Geographic information on logistics routes, including:
        - Key routes
        - Alternate routes and contingency plans implemented to bypass disruptions.
    - Details of coordination efforts with other agents: Qualcomm, Sony Camera, Samsung Display, and LG Chem, SK Hynix, Ibiden and Foxconn Assembly to align logistics with production timelines.
    - Key KPIs, such as delivery time variance, on-time shipment rate, and cost impact of logistical changes.
    - Challenges encountered and the measures taken to resolve them.
    - Recommendations for improving logistics resilience and efficiency in future crises.
    """,
    agent=dhl_logistics
)
tasks.append(task_dhl)

task_amazon = Task(
    description="""
    Amazon must manage the global distribution and sales of the Galaxy S24 Ultra, ensuring timely delivery and customer satisfaction despite the crisis.

    - Assess the current crisis context based on the report provided by crisis_analyst.
    - Coordinate closely with DHL Logistics to align shipping schedules, prioritize key markets, and address disruptions in transportation.
    - Use geographic and logistical insights from DHL Logistics to optimize inventory placement in regional fulfillment centers.
    - Implement customer communication plans to manage expectations and provide updates on order status based on real-time delivery data from DHL Logistics.
    - Report on challenges faced in distribution, such as regional delays or mismatches between supply and demand, and the actions taken to resolve them.
    """,
    expected_output="""
    Amazon Distribution Report – Galaxy S24 Ultra

    Actions Taken:
    - Summary of adjustments made to distribution and sales strategies based on the crisis context.
    - Details of coordination with DHL Logistics, including:
        - Shipping schedule alignment for timely deliveries.
        - Prioritization of key markets based on demand and logistical feasibility.
        - Inventory redistribution to minimize delays in regional fulfillment centers.
    - Implementation of customer communication strategies informed by DHL's real-time delivery updates to manage expectations and improve satisfaction.
    - Key KPIs:
        - Delivery timeliness rate.
        - Inventory turnover rates across regions.
        - Customer satisfaction metrics.
    - Challenges encountered:
        - Regional delays due to logistical bottlenecks reported by DHL Logistics also specify teh geographical positions.
        - Mismatches in demand and supply in certain regions.
    - Solutions Adopted:
        - Dynamic reprioritization of shipments with DHL to address high-priority regions.
        - Enhanced communication channels to keep customers informed about delays and alternatives.
    """,
    agent=amazon_distribution
)
tasks.append(task_amazon)

task_samsung_care = Task(
    description="""
    Samsung Care must provide comprehensive after-sales support for the Galaxy S24 Ultra, ensuring high customer satisfaction during the crisis.

    - Assess the current crisis context based on the report provided by crisis_analyst.
    - Plan and manage spare parts inventory to ensure availability for repairs, considering potential supply chain delays.
    - Adjust repair workflows to maintain efficiency under constrained conditions, prioritizing high-impact cases.
    - Implement proactive customer communication strategies to address concerns and manage expectations.
    - Report on challenges faced in after-sales operations, such as spare part shortages or increased service demand, and the actions taken to address them.
    """,
    expected_output="""
    Samsung Care After-Sales Report – Galaxy S24 Ultra

    Actions Taken:
    - Overview of spare parts inventory management and adjustments made to ensure repair readiness.
    - Details of updated repair workflows to maintain efficiency and prioritize critical cases.
    - Implementation of customer communication strategies to minimize dissatisfaction and build trust.
    - Key KPIs, including repair turnaround time, first-time resolution rate, and customer satisfaction metrics.
    - Challenges encountered (e.g., spare part shortages, increased service demand) and solutions adopted.
    - Recommendations for improving after-sales service resilience and customer engagement in future crises.
    """,
    agent=samsung_care
)
tasks.append(task_samsung_care)

# -----------------------------------------------------------------------------------
# 5.B. ADDITIONAL TASK FOR "SUMMARY AGENT"
# -----------------------------------------------------------------------------------
task_summary = Task(
    description="""
    Summary Agent must collect the most important information from all other agents' outputs 
    and produce a final summary and highlight section.

    - Review the final outputs from each agent (Qualcomm, Samsung Display, Sony Camera, LG Chem, 
      SK Hynix, Ibiden, Foxconn, DHL, Amazon, Samsung Care, and the Crisis Analyst).
    - Extract critical highlights, including major KPIs, challenges, solutions, and overall 
      operational performance.
    - Provide a concise set of bullet points for "Highlights" and a short paragraph summarizing 
      how the crisis was managed.
    """,
    expected_output="""
    Final Consolidated Summary

    Highlights:
    - Key bullet points capturing major achievements and metrics.

    Overall Summary:
    A concise paragraph describing the outcome of the crisis management efforts, 
    including main challenges, solutions, and the final result on supply chain operations.
    """,
    agent=summary_agent
)
tasks.append(task_summary)

# -----------------------------------------------------------------------------------
# 6. CREW SETUP
# -----------------------------------------------------------------------------------
all_agents = [
    crisis_analyst,
    qualcomm_chipset,
    samsung_display,
    sony_camera,
    lg_chem,
    sk_hynix,
    ibiden,
    foxconn_assembly,
    dhl_logistics,
    amazon_distribution,
    samsung_care,
    summary_agent
]

crew = Crew(
    agents=all_agents,
    tasks=tasks,
    process=Process.sequential,
    full_output=True,
    share_crew=False,
)

def get_task_output(task):
    try:
        return task.output.raw
    except AttributeError:
        return "No data available."

def render_agent_report(agent_role, actions, challenges, recommendations):
    st.markdown(f"### {agent_role} Report")
    st.markdown(f"""
    **Actions Taken:**
    {actions}

    **Challenges Encountered:**
    {challenges}

    **Recommendations:**
    {recommendations}
    """)

# -----------------------------------------------------------------------------------
# 7. SIMULATION TITLE & BUTTON (H2, CENTERED, NO EXTRA LINES)
# -----------------------------------------------------------------------------------
with st.container():
    st.markdown("""
    <div style="text-align:center;">
        <h2 style="margin-bottom: 0.5em;">Simulation</h2>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([3.15,1,3])
    with c2:
        run_simulation = st.button("Run Simulation", key="run_sim")

# -----------------------------------------------------------------------------------
# 8. RUN THE SIMULATION
# -----------------------------------------------------------------------------------
if run_simulation:
    # -------------------------------------------------------------------------
    # CRISIS ANALYSIS TASK
    # -------------------------------------------------------------------------
    with st.spinner(""):
        crew.tasks = [task_crisis_analysis]
        crew.kickoff()
    st.markdown("## Crisis Analysis Report")
    st.markdown("---")

    crisis_report = get_task_output(task_crisis_analysis)
    with st.expander("Crisis Analyst Report (Click to Expand)"):
        st.markdown(crisis_report)

    # -------------------------------------------------------------------------
    # PRODUCTION & LOGISTICS TASKS (EXCEPT SUMMARY)
    # -------------------------------------------------------------------------
    with st.spinner(""):
        remaining_tasks = tasks[1:-1]
        for t in remaining_tasks:
            t.description += f"\n\nCrisis Report Details:\n{crisis_report}"
        crew.tasks = remaining_tasks
        crew.kickoff()

    st.markdown("## Production and Logistics Reports")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # SUMMARY AGENT TASK
    # -------------------------------------------------------------------------
    with st.spinner(""):
        final_text = "Below are the outputs from all agents:\n\n"
        for t in remaining_tasks + [task_crisis_analysis]:
            final_text += f"Agent: {t.agent.role} Output:\n"
            final_text += get_task_output(t) + "\n\n"
        task_summary.description += f"\n\nAll Agents' Reports:\n{final_text}"
        crew.tasks = [task_summary]
        crew.kickoff()

    st.markdown("## Final Consolidated Summary")
    st.markdown("---")

    final_summary_output = get_task_output(task_summary)
    with st.expander("Summary Agent's Output (Click to Expand)"):
        st.markdown(final_summary_output)

    # -------------------------------------------------------------------------
    # RESULTS (TABS FOR EACH AGENT EXCEPT SUMMARY)
    # -------------------------------------------------------------------------
    st.markdown("## All Agents' Reports")
    relevant_tasks = [task_crisis_analysis] + remaining_tasks
    tab_labels = [t.agent.role for t in relevant_tasks]
    tabs = st.tabs(tab_labels)

    for i, task_obj in enumerate(relevant_tasks):
        with tabs[i]:
            st.markdown(f"### {task_obj.agent.role} Report")
            with st.expander("Click to view detailed report"):
                st.markdown(get_task_output(task_obj))

    st.markdown("---")
    st.markdown("## End of Simulation")
    st.markdown("Thank you for using the **Supply Chain Simulator for Samsung Galaxy S24 Ultra**!")
