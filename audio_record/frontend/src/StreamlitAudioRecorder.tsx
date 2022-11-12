import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { ReactNode } from "react"

import AudioReactRecorder, { RecordState } from 'audio-react-recorder'
import 'audio-react-recorder/dist/index.css'

interface State {
  isFocused: boolean
  recordState: null
  audioDataURL: string
  reset: boolean
}

class StAudioRec extends StreamlitComponentBase<State> {
  public state = { isFocused: false, recordState: null, audioDataURL: '', reset: false }

  public render = (): ReactNode => {
    const { theme } = this.props
    const style: React.CSSProperties = {}
    const { recordState } = this.state

    if (theme) {
      const borderStyling = `1px solid ${this.state.isFocused ? theme.primaryColor : "gray"}`
      style.border = borderStyling
      style.outline = borderStyling
    }

    return (
      <span>
        <div>
          <button id='record' onClick={this.onClick_start}>
            Start Recording
          </button>
          <button id='stop' onClick={this.onClick_stop}>
            Stop
          </button>
          <button id='reset' onClick={this.onClick_reset}>
            Reset
          </button>

          <button id='continue' onClick={this.onClick_continue}>
            Download
          </button>

          <AudioReactRecorder
            state={recordState}
            onStop={this.onStop_audio}
            type='audio/wav'
            backgroundColor='rgb(255, 255, 255)'
            foregroundColor='rgb(255,76,75)'
            canvasWidth={450}
            canvasHeight={100}
          />

          <audio
            id='audio'
            controls
            src={this.state.audioDataURL}
          />

        </div>
      </span>
    )
  }


  private onClick_start = () => {
    this.setState({
      reset: false,
      audioDataURL: '',
      recordState: RecordState.START
    })
    Streamlit.setComponentValue('')
  }

  private onClick_stop = () => {
    this.setState({
      reset: false,
      recordState: RecordState.STOP
    })
  }

  private onClick_reset = () => {
    this.setState({
      reset: true,
      audioDataURL: '',
      recordState: RecordState.STOP
    })
    Streamlit.setComponentValue('')
  }

  private onClick_continue = () => {
    if (this.state.audioDataURL !== '') {
      // get datetime string for filename
      let datetime = new Date().toLocaleString();
      datetime = datetime.replace(' ', '');
      datetime = datetime.replace(/_/g, '');
      datetime = datetime.replace(',', '');
      var filename = 'streamlit_audio_' + datetime + '.wav';

      // auromatically trigger download
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = this.state.audioDataURL;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
    }
  }

  private onStop_audio = (data) => {
    if (this.state.reset === true) {
      this.setState({
        audioDataURL: ''
      })
      Streamlit.setComponentValue('')
    } else {
      this.setState({
        audioDataURL: data.url
      })

      fetch(data.url).then(function (ctx) {
        return ctx.blob()
      }).then(function (blob) {
        return (new Response(blob)).arrayBuffer()
      }).then(function (buffer) {
        Streamlit.setComponentValue({
          "arr": new Uint8Array(buffer)
        })
      })

    }


  }
}

export default withStreamlitConnection(StAudioRec)

Streamlit.setComponentReady()

Streamlit.setFrameHeight()