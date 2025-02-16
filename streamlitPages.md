# Create a multipage app

In Additional features, we introduced multipage apps, including how to define pages, structure and run multipage apps, and navigate between pages in the user interface. You can read more details in our guide to Multipage apps

## Motivation

Before Streamlit 1.10.0, the streamlit hello command was a large single-page app. As there was no support for multiple pages, we resorted to splitting the app's content using st.selectbox in the sidebar to choose what content to run. The content is comprised of three demos for plotting, mapping, and dataframes.

Here's what the code and single-page app looked like:

### hello.py (👈 Toggle to expand)

Built with Streamlit 🎈
Fullscreen
open_in_new

Notice how large the file is! Each app "page" is written as a function, and the selectbox is used to pick which page to display. As our app grows, maintaining the code requires a lot of additional overhead. Moreover, we're limited by the st.selectbox UI to choose which "page" to run, we cannot customize individual page titles with st.set_page_config, and we're unable to navigate between pages using URLs.

## Convert an existing app into a multipage app

Now that we've identified the limitations of a single-page app, what can we do about it? Armed with our knowledge from the previous section, we can convert the existing app to be a multipage app, of course! At a high level, we need to perform the following steps:

1. Create a new `pages` folder in the same folder where the "entrypoint file" (hello.py) lives
2. Rename our entrypoint file to `Hello.py`, so that the title in the sidebar is capitalized
3. Create three new files inside of pages:
   - `pages/1_📈_Plotting_Demo.py`
   - `pages/2_🌍_Mapping_Demo.py`
   - `pages/3_📊_DataFrame_Demo.py`
4. Move the contents of the plotting_demo, mapping_demo, and data_frame_demo functions into their corresponding new files from Step 3
5. Run `streamlit run Hello.py` to view your newly converted multipage app!

Now, let's walk through each step of the process and view the corresponding changes in code.

## Create the entrypoint file

### Hello.py

```python
import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
```

We rename our entrypoint file to `Hello.py`, so that the title in the sidebar is capitalized and only the code for the intro page is included. Additionally, we're able to customize the page title and favicon — as it appears in the browser tab with st.set_page_config. We can do so for each of our pages too!

Notice how the sidebar does not contain page labels as we haven't created any pages yet.

## Create multiple pages

A few things to remember here:

- We can change the ordering of pages in our MPA by adding numbers to the beginning of each Python file. If we add a 1 to the front of our file name, Streamlit will put that file first in the list.
- The name of each Streamlit app is determined by the file name, so to change the app name you need to change the file name!
- We can add some fun to our app by adding emojis to our file names that will render in our Streamlit app.
- Each page will have its own URL, defined by the name of the file.

Check out how we do all this below! For each new page, we create a new file inside the pages folder, and add the appropriate demo code into it.

### pages/1_📈_Plotting_Demo.py

### pages/2_🌍_Mapping_Demo.py

### pages/3_📊_DataFrame_Demo.py

With our additional pages created, we can now put it all together in the final step below.

## Run the multipage app

To run your newly converted multipage app, run:

```bash
streamlit run Hello.py
```

That's it! The `Hello.py` script now corresponds to the main page of your app, and other scripts that Streamlit finds in the pages folder will also be present in the new page selector that appears in the sidebar.

Built with Streamlit 🎈
Fullscreen
open_in_new

## Next steps

Congratulations! 🎉 If you've read this far, chances are you've learned to create both single-page and multipage apps. Where you go from here is entirely up to your creativity! We're excited to see what you'll build now that adding additional pages to your apps is easier than ever. Try adding more pages to the app we've just built as an exercise. Also, stop by the forum to show off your multipage apps with the Streamlit community! 🎈

Here are a few resources to help you get started:

- Deploy your app for free on Streamlit's Community Cloud.
- Post a question or share your multipage app on our community forum.
- Check out our documentation on Multipage apps.
- Read through Concepts for things like caching, theming, and adding statefulness to apps.
- Browse our API reference for examples of every Streamlit command.