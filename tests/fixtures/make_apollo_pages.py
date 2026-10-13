"""
Writes the three synthetic page captures in tests/fixtures/pages/ (original text about the
Apollo program, laid out the way Chrome's accessibility tree returns real pages:
Wikipedia-style citation marks, "[edit]" links, a pronunciation, an infobox, boilerplate
sections; a Britannica-style page; and a NASA-style page with no heading markup).

    python -m tests.fixtures.make_apollo_pages

Real pages can be captured with tools/capture_page.py instead.
"""

import json
import os

HERE = os.path.join(os.path.dirname(__file__), "pages")

WIKI_TEXT = """Apollo program
From Wikipedia, the free encyclopedia
This article is about the American crewed lunar program. For other uses, see Apollo (disambiguation).
The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which succeeded in landing the first humans on the Moon from 1969 to 1972.[1] It was first conceived during the presidency of Dwight D. Eisenhower as a three-person spacecraft to follow the one-person Project Mercury.[2]
Apollo was later dedicated to President John F. Kennedy's national goal for the 1960s of "landing a man on the Moon and returning him safely to the Earth" in an address to Congress on May 25, 1961.[3] It was the third American human spaceflight program to fly, preceded by the two-person Project Gemini conceived in 1961 to extend spaceflight capability in support of Apollo.[4]
Kennedy's goal was accomplished on the Apollo 11 mission when astronauts Neil Armstrong and Buzz Aldrin landed their Apollo Lunar Module (LM) on July 20, 1969, and walked on the lunar surface, while Michael Collins remained in lunar orbit in the command and service module (CSM).[5] Five subsequent Apollo missions also landed astronauts on the Moon, the last, Apollo 17, in December 1972.[6]
Contents
1 Background
2 Choosing a mission mode
3 Spacecraft
4 Missions
5 Legacy
6 See also
7 References
Background[edit]
Main article: Space Race
The Apollo program was conceived during the Eisenhower administration in early 1960, as a follow-up to Project Mercury.[7] While the Mercury capsule could support only one astronaut on a limited Earth orbital mission, Apollo would carry three.[8] Possible missions included ferrying crews to a space station, circumlunar flights, and eventual crewed lunar landings.[9]
In November 1960, John F. Kennedy was elected president after a campaign that promised American superiority over the Soviet Union in the fields of space exploration and missile defense.[10] The Soviet Union launched the first human, Yuri Gagarin, into orbit on April 12, 1961, which intensified American fears of falling behind.[11]
In 1961, President John F. Kennedy challenged the nation to land an astronaut on the Moon by the end of the decade.[12] The announcement transformed Apollo from a modest research effort into one of the largest engineering projects ever undertaken in peacetime.[13] NASA expanded rapidly, and at its peak the program employed roughly 400,000 people across industry, universities and government.[14]
Choosing a mission mode[edit]
Once Kennedy had defined a goal, the Apollo mission planners were faced with the challenge of designing a spacecraft that could meet it while minimizing risk to human life, cost, and demands on technology and astronaut skill.[15] Four possible mission modes were considered, including direct ascent, Earth orbit rendezvous and lunar surface rendezvous.[16]
In 1962, NASA selected lunar orbit rendezvous, a plan championed by engineer John Houbolt (/ˈhuːboʊlt/) that used a separate lander to descend to the surface.[17] This approach required only one launch of a large rocket and allowed the lander to be built light enough to reach the Moon.[18] Critics initially considered the method too risky because a failed rendezvous in lunar orbit would leave the crew stranded far from Earth.[19]
Spacecraft[edit]
The Apollo spacecraft consisted of a command module, a service module and a lunar module, launched together on top of a Saturn V rocket.[20] The command module was the only part of the spacecraft that returned to Earth, protected by a heat shield during reentry.[21]
The lunar module was a two-stage vehicle designed by Grumman to land two astronauts on the surface and return them to lunar orbit.[22] The Saturn V rocket, developed under the direction of Wernher von Braun at the Marshall Space Flight Center, remains one of the most powerful rockets ever flown.[23]
On January 27, 1967, a cabin fire during a launch rehearsal test killed astronauts Gus Grissom, Ed White and Roger Chaffee.[24] The accident led to a lengthy investigation and to major changes in the design of the command module before crewed flights resumed.[25]
Missions[edit]
Further information: List of Apollo missions
The first crewed flight, Apollo 7, tested the command and service module in Earth orbit in October 1968.[26] Apollo 8 became the first crewed spacecraft to orbit the Moon in December 1968, and its crew photographed the famous Earthrise image.[27]
In July 1969, Apollo 11 achieved the first crewed landing in the Sea of Tranquility.[28] Apollo 13 suffered an oxygen tank explosion in April 1970, forcing the crew to abandon the landing and return safely to Earth using the lunar module as a lifeboat.[29]
The final three missions used the Lunar Roving Vehicle, which allowed astronauts to travel several kilometres from the landing site.[30] In total, twelve astronauts walked on the Moon between 1969 and 1972, and the missions returned 382 kilograms of lunar rock and soil.[31]
Legacy[edit]
The Apollo program stimulated advances in many areas of technology incidental to rocketry and human spaceflight, including avionics, telecommunications and computers.[32] The Apollo Guidance Computer was one of the first computers to use integrated circuits, which helped to drive early demand for the technology.[33]
The program cost about 25.8 billion dollars, a figure that equals well over 200 billion dollars in present-day terms.[34] Images of Earth taken by Apollo crews, such as Earthrise and The Blue Marble, are widely credited with inspiring the environmental movement.[35]
Lunar samples returned by the missions continue to be studied by scientists around the world, and they reshaped theories about the origin of the Moon.[36] Hardware and experience from the program were later used in the Skylab space station and the Apollo-Soyuz Test Project.[37]
See also[edit]
List of Apollo astronauts and the many people who supported the missions from the ground
References[edit]
Benson, Charles D. and William Barnaby Faherty. Moonport: A History of Apollo Launch Facilities and Operations. NASA. Retrieved 12 June 2013. ISBN 978-0-16-048226-0.
Brooks, Courtney G. and others. Chariots for Apollo: A History of Manned Lunar Spacecraft. Archived from the original on 9 February 2008.
"""

BRIT_TEXT = """Apollo
space program
Written by
The Editors of Encyclopaedia Britannica
Last Updated: Sep 3, 2026
Apollo, the United States effort to land astronauts on the Moon, was one of the most ambitious scientific and engineering undertakings of the twentieth century. The program ran from 1961 to 1972 and was managed by the National Aeronautics and Space Administration.
Origins of the program
The program grew out of the political rivalry of the Cold War. After the Soviet Union placed the first satellite and the first human in orbit, American leaders looked for a goal that the United States could reach first. A crewed landing on the Moon was judged to be distant enough that both nations would have to start almost from the beginning.
In May 1961 Kennedy asked Congress to commit the country to a lunar landing before 1970. Congress approved the funding with little debate, and NASA began building new launch facilities in Florida and a mission control centre in Houston.
Technology and spacecraft
Reaching the Moon required a rocket far larger than any that existed at the time. The three-stage Saturn V stood about 110 metres tall and could send roughly 45 tonnes toward the Moon. Its first stage burned kerosene and liquid oxygen, while the upper stages used liquid hydrogen.
The spacecraft itself was assembled from separate modules with different jobs. The command module housed the three-person crew for most of the journey, and the lunar module carried two astronauts down to the surface. Navigation relied on the compact Apollo Guidance Computer working with a sextant and ground tracking.
Lunar landings
Between 1969 and 1972 six missions landed on the Moon at sites chosen for their scientific interest. Astronauts deployed instruments that measured moonquakes, heat flow and the solar wind, and some of these stations returned data for years.
The later missions stayed on the surface for up to three days. Their crews received extensive training in geology so that they could select the most informative rock samples during their excursions.
Scientific results
Analysis of the returned samples showed that the Moon is very old and that its surface was shaped by intense bombardment early in its history. The results supported the idea that the Moon formed from debris after a large body struck the early Earth.
Retroreflectors left on the surface by Apollo crews are still used to measure the distance between Earth and the Moon with laser pulses. These measurements have shown that the Moon is slowly moving away from Earth by a few centimetres each year.
Cite this article
Share
Feedback
Related Topics
Space exploration
Saturn rocket
"""

NASA_TEXT = """Skip to main content
Apollo
The Apollo Missions
The Apollo program brought together hundreds of thousands of engineers, scientists and technicians to achieve a goal that many people had considered impossible only a decade earlier.
Mission control
Flight controllers in Houston monitored every system of the spacecraft around the clock and could send corrections to the crew within seconds. The Mission Operations Control Room became the nerve centre of each flight, from launch until splashdown in the ocean.
During Apollo 13, controllers and engineers worked for four days to improvise procedures that kept the crew alive after the explosion damaged their service module. The safe return of the crew is often described as a successful failure because so much was learned from the emergency.
Astronaut training
Apollo astronauts trained for years before their flights, practising every phase of a mission in simulators at the Kennedy Space Center and the Manned Spacecraft Center. They learned to fly the lunar module using a training vehicle that could imitate the behaviour of a lander in the weak lunar gravity.
Geology field trips took the crews to volcanic regions and impact craters in Arizona, Hawaii and Iceland. Scientists taught them to describe rocks clearly over the radio, so that researchers on Earth could follow their work in real time.
The legacy of Apollo
The program showed that careful engineering and systematic testing could make even very complex systems reliable. Many of the management methods developed for Apollo were later adopted by other large projects in industry and government.
Today NASA is preparing to return astronauts to the Moon through the Artemis program, which builds on the lessons learned during Apollo. The new missions aim to establish a long-term presence near the lunar south pole.
Explore more
Share
Image credit: NASA
"""

PAGES = {
    "apollo_wikipedia": {
        "url": "https://en.wikipedia.org/wiki/Apollo_program",
        "title": "Apollo program - Wikipedia",
        "site": "Wikipedia",
        "accessed": "2026-10-01",
        "main_text": WIKI_TEXT,
        "headings": [
            {"text": "Apollo program", "level": 1}, {"text": "Contents", "level": 2},
            {"text": "Background", "level": 2}, {"text": "Choosing a mission mode", "level": 2},
            {"text": "Spacecraft", "level": 2}, {"text": "Missions", "level": 2},
            {"text": "Legacy", "level": 2}, {"text": "See also", "level": 2}, {"text": "References", "level": 2}],
        "tables": [{"kind": "infobox", "rows": [
            ["Apollo program", "Apollo program"], ["Country", "United States"], ["Organization", "NASA"],
            ["Purpose", "Crewed lunar landing"], ["Status", "Completed"], ["Program history", "Program history"],
            ["Cost", "$25.8 billion (1973)"], ["Duration", "1961–1972"],
            ["First flight", "February 26, 1966[1]"], ["First crewed flight", "October 11, 1968"],
            ["Last flight", "December 7, 1972"], ["Launch site(s)", "Kennedy Space Center"],
            ["Vehicle information", "Vehicle information"],
            ["Crew vehicle", "Apollo command and service module"],
            ["Launch vehicle(s)", "Saturn IB and Saturn V"]]}],
        "seen_intervals": [[0, 22], [18, 41], [35, 63], [58, 80]],
        "method": "main-landmark",
    },
    "apollo_britannica": {
        "url": "https://www.britannica.com/science/Apollo-space-program",
        "title": "Apollo | History, Missions, Significance, & Facts | Britannica",
        "site": "Encyclopaedia Britannica",
        "accessed": "2026-10-01",
        "main_text": BRIT_TEXT,
        "headings": [{"text": "Apollo", "level": 1}, {"text": "Origins of the program", "level": 2},
                     {"text": "Technology and spacecraft", "level": 2}, {"text": "Lunar landings", "level": 2},
                     {"text": "Scientific results", "level": 2}, {"text": "Related Topics", "level": 2}],
        "tables": [],
        "seen_intervals": [],
        "method": "article",
    },
    "apollo_nasa": {
        "url": "https://www.nasa.gov/the-apollo-program/",
        "title": "The Apollo Program - NASA",
        "site": "NASA",
        "accessed": "2026-10-01",
        "main_text": NASA_TEXT,
        "headings": [],
        "tables": [],
        "seen_intervals": [[0, 45], [40, 100]],
        "method": "document",
    },
}


def main():
    """Write each page in PAGES to tests/fixtures/pages/<name>.json (with its word count)."""
    os.makedirs(HERE, exist_ok=True)
    for name, cap in PAGES.items():
        cap = dict(cap, total_words=len(cap["main_text"].split()))
        with open(os.path.join(HERE, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(cap, f, indent=1, ensure_ascii=False)
        print(f"wrote {name}.json")


if __name__ == "__main__":
    main()
