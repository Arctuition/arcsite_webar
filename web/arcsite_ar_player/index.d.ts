/** Opens USDZ models in AR Quick Look on iPhone and iPad, at true size (scaling disabled). */
export declare const ArPlayer: {
  /** Whether AR Quick Look can open here. Never throws. */
  checkSupport(): boolean;

  /**
   * Opens the model at url. Call it straight from a tap handler, with no await before it.
   * @throws Error where Quick Look is not supported; TypeError for an empty or invalid URL.
   */
  launchUrl(url: string): void;

  /**
   * Opens a USDZ file from its bytes. Call it straight from a tap handler, with no await before it.
   * @throws Error where Quick Look is not supported; TypeError if the bytes are not a USDZ file.
   */
  launchBytes(usdz: Uint8Array): void;
};
